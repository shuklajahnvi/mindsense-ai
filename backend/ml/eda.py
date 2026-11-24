"""Exploratory data analysis for GoEmotions and Dreaddit.

Usage:  python -m ml.eda [--task emotion|stress|all]
Outputs JSON summaries and PNG figures under artifacts/reports/<task>/.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ml.data import load_dreaddit, load_goemotions  # noqa: E402
from ml.paths import report_dir  # noqa: E402


def _length_stats(series) -> dict:
    lens = series.str.split().map(len)
    return {"mean_words": float(lens.mean()), "median_words": float(lens.median()),
            "p95_words": float(lens.quantile(0.95)), "max_words": int(lens.max())}


def emotion_eda() -> dict:
    s = load_goemotions()
    out = report_dir("emotion")
    names = s.label_names
    summary = {"dataset": "GoEmotions (simplified)", "n_labels": len(names), "labels": names, "splits": {}}
    for split_name, df in (("train", s.train), ("validation", s.val), ("test", s.test)):
        y = s.matrix(df)
        card = y.sum(1)
        summary["splits"][split_name] = {
            "n": int(len(df)),
            "label_counts": {n: int(c) for n, c in zip(names, y.sum(0))},
            "labels_per_example": {str(k): int(v) for k, v in sorted(Counter(card.astype(int)).items())},
            "mean_labels_per_example": float(card.mean()),
            "length": _length_stats(df["clean"]),
        }
    ytr = s.matrix(s.train)
    co = (ytr.T @ ytr).astype(int)
    np.fill_diagonal(co, 0)
    pairs = sorted(((co[i, j], names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))),
                   reverse=True)[:15]
    summary["top_cooccurring_pairs_train"] = [{"a": a, "b": b, "count": int(c)} for c, a, b in pairs]
    counts = summary["splits"]["train"]["label_counts"]
    summary["imbalance_ratio_max_min"] = max(counts.values()) / max(1, min(counts.values()))

    order = sorted(counts, key=counts.get, reverse=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(order, [counts[k] for k in order], color="#6366f1")
    ax.set_title("GoEmotions train: examples per label")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout(); fig.savefig(out / "eda_label_distribution.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(s.train["clean"].str.split().map(len), bins=40, color="#0ea5e9")
    ax.set_title("GoEmotions train: words per comment"); fig.tight_layout()
    fig.savefig(out / "eda_length_hist.png", dpi=120); plt.close(fig)

    (out / "eda.json").write_text(json.dumps(summary, indent=2))
    return summary


def stress_eda() -> dict:
    s = load_dreaddit()
    out = report_dir("stress")
    summary = {"dataset": "Dreaddit (labelled subset)", "leakage_checks": s.leakage_report, "splits": {}}
    for split_name, df in (("train", s.train), ("validation", s.val), ("test", s.test)):
        summary["splits"][split_name] = {
            "n": int(len(df)),
            "n_posts": int(df["post_id"].nunique()),
            "label_counts": {"not_stress": int((df.label == 0).sum()), "stress": int((df.label == 1).sum())},
            "stress_rate": float(df.label.mean()),
            "subreddits": {k: int(v) for k, v in df["subreddit"].value_counts().items()},
            "stress_rate_by_subreddit": {k: float(v) for k, v in df.groupby("subreddit")["label"].mean().items()},
            "mean_annotator_confidence": float(df["confidence"].mean()),
            "length": _length_stats(df["clean"]),
        }
    tr = s.train
    sub = tr.groupby("subreddit")["label"].agg(["mean", "count"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(sub.index, sub["mean"], color="#f97316")
    ax.set_xlabel("share labelled stress"); ax.set_title("Dreaddit train: stress label rate by subreddit")
    fig.tight_layout(); fig.savefig(out / "eda_subreddit_rate.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for lab, col in ((0, "#22c55e"), (1, "#ef4444")):
        ax.hist(tr[tr.label == lab]["clean"].str.split().map(len), bins=30, alpha=0.6, color=col,
                label=["not stress", "stress"][lab])
    ax.legend(); ax.set_title("Dreaddit train: words per segment"); fig.tight_layout()
    fig.savefig(out / "eda_length_hist.png", dpi=120); plt.close(fig)

    (out / "eda.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["emotion", "stress", "all"], default="all")
    a = ap.parse_args()
    if a.task in ("emotion", "all"):
        e = emotion_eda()
        print("GoEmotions:", {k: v["n"] for k, v in e["splits"].items()},
              "mean labels/example:", round(e["splits"]["train"]["mean_labels_per_example"], 3))
    if a.task in ("stress", "all"):
        st = stress_eda()
        print("Dreaddit:", {k: (v["n"], round(v["stress_rate"], 3)) for k, v in st["splits"].items()},
              "leakage:", st["leakage_checks"])
