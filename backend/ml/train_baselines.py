"""Traditional ML baselines.

* Emotion: TF-IDF (word 1-2 grams) + One-vs-Rest Logistic Regression,
  i.e. 28 independent binary classifiers (correct multi-label formulation),
  per-class thresholds tuned on the validation split.
* Stress:  TF-IDF + Logistic Regression and TF-IDF + linear SVM
  (Platt-calibrated so it yields probabilities for ROC-AUC and LIME).

Hyper-parameters are chosen on validation only; the test split is used once.

Usage:  python -m ml.train_baselines [--task emotion|stress|all] [--train-limit N]
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.common.io import to_jsonable, write_meta
from ml.common.metrics import binary_report, multilabel_report, tune_thresholds
from ml.common.text import normalize_for_tfidf
from ml.data import load_dreaddit, load_goemotions
from ml.paths import model_dir


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(preprocessor=normalize_for_tfidf, ngram_range=(1, 2), min_df=2, max_df=0.9,
                           sublinear_tf=True, max_features=100_000, token_pattern=r"(?u)\b\w[\w']*\b")


def train_emotion_baseline(train_limit: int | None = None) -> dict:
    s = load_goemotions(train_limit=train_limit)
    ytr, yva, yte = s.matrix(s.train), s.matrix(s.val), s.matrix(s.test)
    best = None
    for C in (1.0, 4.0, 10.0):
        pipe = Pipeline([("tfidf", _tfidf()),
                         ("clf", OneVsRestClassifier(
                             LogisticRegression(C=C, max_iter=2000, class_weight="balanced", solver="liblinear")))])
        t0 = time.time()
        # Sequential fit: process workers get read-only memmaps (liblinear rejects them) and threads race on
        # liblinear's in-place index sorting of the shared sparse matrix.
        pipe.fit(s.train["clean"], ytr)
        pva = pipe.predict_proba(s.val["clean"])
        th = tune_thresholds(yva, pva)
        val = multilabel_report(yva, pva, th, s.label_names)
        print(f"[emotion-baseline] C={C}: val macro-F1={val['macro_f1']:.4f} micro-F1={val['micro_f1']:.4f} "
              f"({time.time() - t0:.0f}s)")
        if best is None or val["macro_f1"] > best[2]["macro_f1"]:
            best = (C, pipe, val, th)
    C, pipe, val, th = best
    test = multilabel_report(yte, pipe.predict_proba(s.test["clean"]), th, s.label_names)
    out = model_dir("emotion_baseline")
    joblib.dump(pipe, out / "model.joblib")
    meta = {
        "name": "emotion_baseline", "task": "emotion", "kind": "baseline",
        "algorithm": "TF-IDF (1-2 grams) + One-vs-Rest Logistic Regression (class_weight=balanced)",
        "fine_tuned": False, "labels": s.label_names, "thresholds": th.tolist(),
        "hyperparameters": {"C": C},
        "training": {"dataset": "GoEmotions simplified", "n_train": len(s.train), "n_val": len(s.val),
                     "n_test": len(s.test), "train_limit": train_limit},
        "metrics": {"validation": val, "test": test},
    }
    write_meta(out, to_jsonable(meta))
    print(f"[emotion-baseline] TEST macro-F1={test['macro_f1']:.4f} micro-F1={test['micro_f1']:.4f}")
    return meta


def train_stress_baselines() -> list[dict]:
    s = load_dreaddit()
    ytr, yva, yte = (s.train.label.values, s.val.label.values, s.test.label.values)
    results = []
    candidates = {
        "stress_baseline_lr": ("TF-IDF (1-2 grams) + Logistic Regression",
                               lambda C: LogisticRegression(C=C, max_iter=3000, solver="liblinear")),
        "stress_baseline_svm": ("TF-IDF (1-2 grams) + Linear SVM (sigmoid-calibrated)",
                                lambda C: CalibratedClassifierCV(LinearSVC(C=C), method="sigmoid", cv=5)),
    }
    for name, (algo, make) in candidates.items():
        best = None
        for C in (0.1, 0.5, 1.0, 4.0, 10.0):
            pipe = Pipeline([("tfidf", _tfidf()), ("clf", make(C))])
            pipe.fit(s.train["clean"], ytr)
            val = binary_report(yva, pipe.predict_proba(s.val["clean"])[:, 1])
            print(f"[{name}] C={C}: val F1={val['f1']:.4f} AUC={val['roc_auc']:.4f}")
            if best is None or val["f1"] > best[2]["f1"]:
                best = (C, pipe, val)
        C, pipe, val = best
        test = binary_report(yte, pipe.predict_proba(s.test["clean"])[:, 1])
        out = model_dir(name)
        joblib.dump(pipe, out / "model.joblib")
        meta = {
            "name": name, "task": "stress", "kind": "baseline", "algorithm": algo, "fine_tuned": False,
            "labels": ["not_stress", "stress"], "threshold": 0.5, "hyperparameters": {"C": C},
            "training": {"dataset": "Dreaddit", "n_train": len(s.train), "n_val": len(s.val),
                         "n_test": len(s.test), "leakage_checks": s.leakage_report},
            "metrics": {"validation": val, "test": test},
        }
        write_meta(out, to_jsonable(meta))
        print(f"[{name}] TEST acc={test['accuracy']:.4f} F1={test['f1']:.4f} AUC={test['roc_auc']:.4f}")
        results.append(meta)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["emotion", "stress", "all"], default="all")
    ap.add_argument("--train-limit", type=int, default=None, help="dev subset size for GoEmotions train")
    a = ap.parse_args()
    np.random.seed(42)
    if a.task in ("stress", "all"):
        train_stress_baselines()
    if a.task in ("emotion", "all"):
        train_emotion_baseline(a.train_limit)
