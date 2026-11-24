"""Loading and running the trained emotion and stress classifiers.

Models are produced by the scripts in ``backend/ml``. At runtime we prefer the
fine-tuned transformer and fall back to the TF-IDF baseline. If neither has
been trained, the classifier reports itself as unavailable - predictions are
never invented.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import get_settings
from ml.common.text import clean_text
from ml.paths import MODELS_DIR

EMOTION_DIRS = {"transformer": "emotion_transformer", "baseline": "emotion_baseline"}
STRESS_DIRS = {"transformer": "stress_transformer", "baseline": "stress_baseline_lr"}


def _read_meta(path: Path) -> dict | None:
    f = path / "meta.json"
    return json.loads(f.read_text()) if f.exists() else None


@dataclass
class LoadedModel:
    task: str
    variant: str          # transformer | baseline
    meta: dict
    predict_proba: callable  # list[str] -> np.ndarray (n, n_labels)

    @property
    def labels(self) -> list[str]:
        return self.meta["labels"]

    def info(self) -> dict:
        t = self.meta.get("metrics", {}).get("test", {})
        return {
            "name": self.meta["name"], "variant": self.variant, "algorithm": self.meta["algorithm"],
            "fine_tuned": bool(self.meta.get("fine_tuned")), "base_checkpoint": self.meta.get("base_checkpoint"),
            "trained_on_subset": bool(self.meta.get("training", {}).get("is_subset")),
            "n_train": self.meta.get("training", {}).get("n_train"),
            "test_headline": {k: t.get(k) for k in ("macro_f1", "micro_f1", "f1", "accuracy", "roc_auc") if k in t},
        }


def _load_baseline(path: Path, meta: dict, task: str) -> LoadedModel:
    import joblib

    pipe = joblib.load(path / "model.joblib")

    def predict_proba(texts: list[str]) -> np.ndarray:
        return np.asarray(pipe.predict_proba([clean_text(t) for t in texts]))

    return LoadedModel(task, "baseline", meta, predict_proba)


def _load_transformer(path: Path, meta: dict, task: str) -> LoadedModel:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    max_len = int(meta.get("max_len", 128))
    lock = threading.Lock()

    def windows(text: str) -> list[str]:
        """Journal entries are often longer than the training texts (GoEmotions comments, max_len 64).
        Rather than truncating, pack whole sentences into windows that fit the model's max_len."""
        budget = max_len - 2
        if len(tok(text, add_special_tokens=False)["input_ids"]) <= budget:
            return [text]
        out, cur, cur_len = [], [], 0
        for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
            if not sent.strip():
                continue
            n = len(tok(sent, add_special_tokens=False)["input_ids"])
            if cur and cur_len + n > budget:
                out.append(" ".join(cur)); cur, cur_len = [], 0
            cur.append(sent); cur_len += n
        if cur:
            out.append(" ".join(cur))
        return out or [text]

    @torch.no_grad()
    def predict_proba(texts: list[str], batch_size: int = 32) -> np.ndarray:
        segs, owner = [], []
        for i, t in enumerate(texts):
            for w in windows(clean_text(t)):
                segs.append(w); owner.append(i)
        out = []
        with lock:
            for i in range(0, len(segs), batch_size):
                enc = tok(segs[i:i + batch_size], truncation=True, max_length=max_len, padding=True,
                          return_tensors="pt")
                logits = model(**enc).logits
                out.append((torch.sigmoid(logits) if task == "emotion" else torch.softmax(logits, -1)).numpy())
        probs, owner = np.concatenate(out), np.asarray(owner)
        # Emotion: an emotion expressed in any window counts (max). Stress: overall wording (mean).
        agg = np.max if task == "emotion" else np.mean
        return np.stack([agg(probs[owner == i], axis=0) for i in range(len(texts))])

    return LoadedModel(task, "transformer", meta, predict_proba)


class ModelRegistry:
    def __init__(self):
        self._cache: dict[tuple[str, str], LoadedModel] = {}
        self._lock = threading.Lock()

    @staticmethod
    def available(task: str) -> dict[str, dict | None]:
        dirs = EMOTION_DIRS if task == "emotion" else STRESS_DIRS
        return {v: _read_meta(MODELS_DIR / d) for v, d in dirs.items()}

    def resolve_variant(self, task: str, requested: str | None = None) -> str | None:
        pref = requested or (get_settings().emotion_model if task == "emotion" else get_settings().stress_model)
        avail = {k for k, v in self.available(task).items() if v}
        if pref in avail:
            return pref
        for v in ("transformer", "baseline"):
            if v in avail:
                return v
        return None

    def get(self, task: str, variant: str | None = None) -> LoadedModel | None:
        v = self.resolve_variant(task, variant)
        if v is None:
            return None
        key = (task, v)
        with self._lock:
            if key not in self._cache:
                dirs = EMOTION_DIRS if task == "emotion" else STRESS_DIRS
                path = MODELS_DIR / dirs[v]
                meta = _read_meta(path)
                loader = _load_transformer if v == "transformer" else _load_baseline
                self._cache[key] = loader(path, meta, task)
            return self._cache[key]

    def reload(self) -> None:
        with self._lock:
            self._cache.clear()


registry = ModelRegistry()


def predict_emotions(text: str, variant: str | None = None) -> tuple[list[dict], dict] | None:
    m = registry.get("emotion", variant)
    if m is None:
        return None
    probs = m.predict_proba([text])[0]
    th = np.asarray(m.meta.get("thresholds", [0.5] * len(m.labels)))
    rows = [{"label": lab, "score": float(p), "threshold": float(t), "predicted": bool(p >= t)}
            for lab, p, t in zip(m.labels, probs, th)]
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows, m.info()


def predict_stress(text: str, variant: str | None = None) -> tuple[dict, dict] | None:
    m = registry.get("stress", variant)
    if m is None:
        return None
    p = float(m.predict_proba([text])[0][1])
    th = float(m.meta.get("threshold", 0.5))
    return {"label": "stress" if p >= th else "not_stress", "score": p, "threshold": th}, m.info()
