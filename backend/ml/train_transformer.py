"""Fine-tune a pretrained transformer for emotion (multi-label) or stress (binary).

* Emotion: sigmoid outputs + ``BCEWithLogitsLoss`` over 28 labels; per-class
  thresholds tuned on validation after training.
* Stress: 2-way classification head with cross-entropy (binary task, softmax
  over {not_stress, stress} is the correct formulation here).

Designed to be CPU-friendly (no GPU on the development machine) but uses CUDA
automatically when available. Supports dev subsets, configurable batch size,
gradient accumulation, length-bucketed dynamic padding and early stopping.

Examples
--------
  python -m ml.train_transformer --task stress  --epochs 4
  python -m ml.train_transformer --task emotion --epochs 2 --train-limit 12000   # dev subset
  python -m ml.train_transformer --task emotion --epochs 3                       # full research run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ml.common.io import to_jsonable, write_meta
from ml.common.metrics import binary_report, multilabel_report, tune_thresholds
from ml.data import load_dreaddit, load_goemotions
from ml.paths import model_dir, report_dir

DEFAULTS = {
    "emotion": {"model": "distilroberta-base", "max_len": 64, "batch_size": 32, "lr": 5e-5, "epochs": 3},
    "stress": {"model": "distilroberta-base", "max_len": 192, "batch_size": 16, "lr": 3e-5, "epochs": 4},
}


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def bucketed_batches(lengths: list[int], batch_size: int, shuffle: bool, seed: int) -> list[list[int]]:
    """Group similar-length examples to minimise padding (big win on CPU)."""
    idx = np.arange(len(lengths))
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(idx)
        chunks = [idx[i:i + batch_size * 50] for i in range(0, len(idx), batch_size * 50)]
        idx = np.concatenate([c[np.argsort([lengths[j] for j in c])] for c in chunks])
    else:
        idx = idx[np.argsort(lengths)]
    batches = [idx[i:i + batch_size].tolist() for i in range(0, len(idx), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


class Encoded:
    def __init__(self, tokenizer, texts, labels, max_len):
        enc = tokenizer(list(texts), truncation=True, max_length=max_len)
        self.input_ids = enc["input_ids"]
        self.labels = labels
        self.lengths = [len(x) for x in self.input_ids]
        self.pad = tokenizer.pad_token_id

    def collate(self, batch_idx):
        L = max(self.lengths[i] for i in batch_idx)
        ids = torch.full((len(batch_idx), L), self.pad, dtype=torch.long)
        mask = torch.zeros((len(batch_idx), L), dtype=torch.long)
        for r, i in enumerate(batch_idx):
            seq = self.input_ids[i]
            ids[r, :len(seq)] = torch.tensor(seq)
            mask[r, :len(seq)] = 1
        return ids, mask, torch.as_tensor(self.labels[batch_idx])


@torch.no_grad()
def predict_probs(model, data: Encoded, task: str, device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for b in bucketed_batches(data.lengths, batch_size, shuffle=False, seed=0):
        ids, mask, _ = data.collate(b)
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        p = torch.sigmoid(logits) if task == "emotion" else torch.softmax(logits, -1)[:, 1:2]
        out.append((b, p.cpu().numpy()))
    probs = np.zeros((len(data.lengths), out[0][1].shape[1]), dtype=np.float32)
    for b, p in out:
        probs[b] = p
    return probs if task == "emotion" else probs[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["emotion", "stress"], required=True)
    ap.add_argument("--model", default=None, help="Hugging Face checkpoint to fine-tune")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--train-limit", type=int, default=None, help="dev subset of training examples")
    ap.add_argument("--eval-limit", type=int, default=None, help="dev subset of validation examples")
    ap.add_argument("--patience", type=int, default=2, help="early-stopping patience (evaluations)")
    ap.add_argument("--evals-per-epoch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="freeze the embedding layer (big CPU speed-up: removes ~38M params from AdamW)")
    ap.add_argument("--freeze-layers", type=int, default=0, help="freeze the bottom N transformer layers")
    ap.add_argument("--head-lr", type=float, default=None, help="learning rate for the classification head")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="output model name (default <task>_transformer)")
    a = ap.parse_args()

    cfg = {**DEFAULTS[a.task], **{k: v for k, v in {
        "model": a.model, "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr, "max_len": a.max_len
    }.items() if v is not None}}
    name = a.name or f"{a.task}_transformer"
    set_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{name}] device={device} threads={torch.get_num_threads()} cfg={cfg}", flush=True)

    # ---------------- data ----------------
    if a.task == "emotion":
        s = load_goemotions(train_limit=a.train_limit, eval_limit=a.eval_limit)
        labels = s.label_names
        tr_y, va_y, te_y = s.matrix(s.train), s.matrix(s.val), s.matrix(s.test)
        tr_x, va_x, te_x = s.train["clean"], s.val["clean"], s.test["clean"]
        extra = {}
    else:
        s = load_dreaddit()
        labels = ["not_stress", "stress"]
        tr_y, va_y, te_y = (s.train.label.values.astype(np.int64), s.val.label.values.astype(np.int64),
                            s.test.label.values.astype(np.int64))
        tr_x, va_x, te_x = s.train["clean"], s.val["clean"], s.test["clean"]
        if a.train_limit:
            tr_x, tr_y = tr_x[:a.train_limit], tr_y[:a.train_limit]
        extra = {"leakage_checks": s.leakage_report}

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    train_d = Encoded(tok, tr_x, tr_y, cfg["max_len"])
    val_d = Encoded(tok, va_x, va_y, cfg["max_len"])
    test_d = Encoded(tok, te_x, te_y, cfg["max_len"])
    print(f"[{name}] n_train={len(tr_x)} n_val={len(va_x)} n_test={len(te_x)}", flush=True)

    # ---------------- model ----------------
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], num_labels=len(labels),
        problem_type="multi_label_classification" if a.task == "emotion" else "single_label_classification",
        id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)},
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss() if a.task == "emotion" else nn.CrossEntropyLoss()

    # Optional partial fine-tuning (for CPU-only training budgets).
    base = getattr(model, model.base_model_prefix)
    if a.freeze_embeddings:
        for p in base.embeddings.parameters():
            p.requires_grad = False
    if a.freeze_layers:
        for layer in base.encoder.layer[:a.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
    n_layers = len(base.encoder.layer)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    scope = ("full fine-tuning" if not (a.freeze_embeddings or a.freeze_layers) else
             f"partial fine-tuning: top {n_layers - a.freeze_layers} of {n_layers} transformer layers + "
             f"classification head{'' if a.freeze_embeddings else ' + embeddings'} trained"
             f"{'; embeddings frozen' if a.freeze_embeddings else ''}"
             f"{f'; bottom {a.freeze_layers} layers frozen' if a.freeze_layers else ''}")
    print(f"[{name}] {scope}; trainable params {trainable / 1e6:.1f}M / {total / 1e6:.1f}M", flush=True)

    # Multi-label: initialise the output bias to each label's prior log-odds so training does not start
    # in the "predict everything negative" regime (labels are sparse: ~1.2 of 28 positive per example).
    head_out = [m for m in model.classifier.modules() if isinstance(m, nn.Linear)][-1]
    if a.task == "emotion":
        prior = np.clip(tr_y.mean(0), 1e-4, 1 - 1e-4)
        with torch.no_grad():
            head_out.bias.copy_(torch.tensor(np.log(prior / (1 - prior)), dtype=head_out.bias.dtype))

    steps_per_epoch = math.ceil(len(train_d.lengths) / cfg["batch_size"] / a.grad_accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    # Randomly initialised classification head gets a higher learning rate than the pretrained encoder.
    head_params = [p for n, p in model.named_parameters() if n.startswith("classifier") and p.requires_grad]
    body_params = [p for n, p in model.named_parameters() if not n.startswith("classifier") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": body_params, "lr": cfg["lr"]},
                             {"params": head_params, "lr": a.head_lr or cfg["lr"]}], weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    eval_every = max(1, math.ceil(len(train_d.lengths) / cfg["batch_size"] / a.evals_per_epoch))

    def evaluate() -> tuple[float, dict]:
        p = predict_probs(model, val_d, a.task, device)
        if a.task == "emotion":
            rep = multilabel_report(va_y, p, tune_thresholds(va_y, p), labels)
            return rep["macro_f1"], rep
        rep = binary_report(va_y, p)
        return rep["f1"], rep

    out = model_dir(name)
    history, best_score, bad_evals, stop = [], -1.0, 0, False
    t_start, step, batch_no = time.time(), 0, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        batches = bucketed_batches(train_d.lengths, cfg["batch_size"], shuffle=True, seed=a.seed + epoch)
        running = 0.0
        for i, b in enumerate(batches):
            ids, mask, y = train_d.collate(b)
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits, y.to(device)) / a.grad_accum
            loss.backward()
            running += loss.item() * a.grad_accum
            batch_no += 1
            if batch_no % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
            if (i + 1) % 50 == 0:
                el = time.time() - t_start
                print(f"[{name}] epoch {epoch + 1} batch {i + 1}/{len(batches)} loss={running / (i + 1):.4f} "
                      f"elapsed={el / 60:.1f}m", flush=True)
            if (i + 1) % eval_every == 0 or i + 1 == len(batches):
                score, rep = evaluate()
                model.train()
                history.append({"epoch": epoch + 1, "batch": i + 1, "train_loss": running / (i + 1),
                                "val_score": score, "elapsed_min": (time.time() - t_start) / 60})
                print(f"[{name}] EVAL epoch {epoch + 1} batch {i + 1}: val "
                      f"{'macro-F1' if a.task == 'emotion' else 'F1'}={score:.4f}", flush=True)
                if score > best_score:
                    best_score, bad_evals = score, 0
                    model.save_pretrained(out); tok.save_pretrained(out)
                else:
                    bad_evals += 1
                    if bad_evals >= a.patience:
                        print(f"[{name}] early stopping", flush=True)
                        stop = True
                        break
        if stop:
            break
    train_minutes = (time.time() - t_start) / 60

    # ---------------- final evaluation with best checkpoint ----------------
    model = AutoModelForSequenceClassification.from_pretrained(out).to(device)
    pva, pte = predict_probs(model, val_d, a.task, device), predict_probs(model, test_d, a.task, device)
    if a.task == "emotion":
        th = tune_thresholds(va_y, pva)
        val, test = multilabel_report(va_y, pva, th, labels), multilabel_report(te_y, pte, th, labels)
        thresholds = {"thresholds": th.tolist()}
        headline = f"macro-F1={test['macro_f1']:.4f} micro-F1={test['micro_f1']:.4f}"
    else:
        val, test = binary_report(va_y, pva), binary_report(te_y, pte)
        thresholds = {"threshold": 0.5}
        headline = f"acc={test['accuracy']:.4f} F1={test['f1']:.4f} AUC={test['roc_auc']:.4f}"
    meta = {
        "name": name, "task": a.task, "kind": "transformer",
        "algorithm": f"Fine-tuned {cfg['model']} "
                     f"({'sigmoid + BCEWithLogitsLoss, multi-label' if a.task == 'emotion' else 'softmax + cross-entropy, binary'})",
        "fine_tuned": True, "base_checkpoint": cfg["model"], "labels": labels, **thresholds,
        "max_len": cfg["max_len"], "fine_tuning_scope": scope,
        "trainable_parameters": trainable, "total_parameters": total,
        "hyperparameters": {**cfg, "grad_accum": a.grad_accum, "patience": a.patience, "seed": a.seed,
                            "freeze_embeddings": a.freeze_embeddings, "freeze_layers": a.freeze_layers,
                            "head_lr": a.head_lr, "head_bias_init": "label prior log-odds" if a.task == "emotion" else None},
        "training": {"dataset": "GoEmotions simplified" if a.task == "emotion" else "Dreaddit",
                     "n_train": len(tr_x), "n_val": len(va_x), "n_test": len(te_x),
                     "train_limit": a.train_limit, "eval_limit": a.eval_limit,
                     "is_subset": bool(a.train_limit), "device": str(device),
                     "optimizer_steps": step, "train_minutes": round(train_minutes, 1),
                     "history": history, **extra},
        "metrics": {"validation": val, "test": test},
    }
    write_meta(out, to_jsonable(meta))
    (report_dir(a.task) / f"{name}_history.json").write_text(json.dumps(history, indent=2))
    print(f"[{name}] DONE in {train_minutes:.1f} min. TEST {headline}", flush=True)


if __name__ == "__main__":
    main()
