"""Evaluation helpers for multi-label (emotion) and binary (stress) tasks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def tune_thresholds(y_true: np.ndarray, probs: np.ndarray, grid=None) -> np.ndarray:
    """Per-class decision thresholds maximising F1 on a *validation* set."""
    grid = np.arange(0.05, 0.95, 0.025) if grid is None else grid
    thresholds = np.full(y_true.shape[1], 0.5)
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() == 0:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f = f1_score(y_true[:, j], (probs[:, j] >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds[j] = float(best_t)
    return thresholds


def multilabel_report(y_true: np.ndarray, probs: np.ndarray, thresholds, label_names: list[str]) -> dict:
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=float), (y_true.shape[1],))
    y_pred = (probs >= thresholds).astype(int)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    per_class = {
        name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i]),
               "threshold": float(thresholds[i])}
        for i, name in enumerate(label_names)
    }
    try:
        macro_auc = float(roc_auc_score(y_true, probs, average="macro"))
    except ValueError:
        macro_auc = None
    # Pairwise "confusion": how often the model predicts label B when the gold label is A only.
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    confusion = {
        name: {"tn": int(m[0, 0]), "fp": int(m[0, 1]), "fn": int(m[1, 0]), "tp": int(m[1, 1])}
        for name, m in zip(label_names, mcm)
    }
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "macro_roc_auc": macro_auc,
        "n_samples": int(y_true.shape[0]),
        "per_class": per_class,
        "per_class_confusion": confusion,
        "label_cooccurrence_errors": top_confusions(y_true, y_pred, label_names),
    }


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str], k: int = 15) -> list[dict]:
    """Most frequent (gold label -> wrongly predicted label) pairs.

    For each example and each gold label A that is present, every predicted
    label B that is *not* gold counts as an A->B confusion.
    """
    n = len(label_names)
    counts = np.zeros((n, n), dtype=int)
    for t, pr in zip(y_true, y_pred):
        gold = np.where(t == 1)[0]
        wrong = np.where((pr == 1) & (t == 0))[0]
        for a in gold:
            for b in wrong:
                counts[a, b] += 1
    pairs = [(counts[a, b], a, b) for a in range(n) for b in range(n) if counts[a, b] > 0]
    pairs.sort(reverse=True)
    return [{"gold": label_names[a], "predicted": label_names[b], "count": int(c)} for c, a, b in pairs[:k]]


def binary_report(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (prob_pos >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob_pos)),
        "threshold": float(threshold),
        "confusion_matrix": {"labels": ["not_stress", "stress"], "matrix": cm.tolist()},
        "n_samples": int(len(y_true)),
    }
