"""Longitudinal analysis of stored journal entries.

All statements are descriptive and refer to *language patterns in the
entries*, never to the user's clinical state. Changes in model scores are not
interpreted as improvement or deterioration of mental health.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Sentiment grouping of GoEmotions labels as defined by Demszky et al. (2020).
EMOTION_GROUPS = {
    "positive": ["admiration", "amusement", "approval", "caring", "desire", "excitement", "gratitude", "joy",
                 "love", "optimism", "pride", "relief"],
    "negative": ["anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear",
                 "grief", "nervousness", "remorse", "sadness"],
    "ambiguous": ["confusion", "curiosity", "realization", "surprise"],
    "neutral": ["neutral"],
}
LABEL_TO_GROUP = {lab: g for g, labs in EMOTION_GROUPS.items() for lab in labs}
CAVEAT = ("Trends summarise model scores for the language in your entries. They are not a measure of your "
          "mental health and a change in a score is not a clinical improvement or deterioration.")


def to_frame(entries) -> pd.DataFrame:
    rows = []
    for e in entries:
        emo = {r["label"]: r["score"] for r in (e.emotions or [])}
        rows.append({
            "id": e.id, "created_at": e.created_at, "stress_score": e.stress_score,
            "stress_flag": None if e.stress_label is None else int(e.stress_label == "stress"),
            "mood": e.mood, "energy": e.energy, "sleep_hours": e.sleep_hours, "sleep_quality": e.sleep_quality,
            "predicted": list(e.predicted_emotions or []), "emo": emo,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.normalize()
    return df.sort_values("created_at").reset_index(drop=True)


def _mean(s: pd.Series):
    s = s.dropna()
    return None if s.empty else round(float(s.mean()), 4)


def _group_scores(emo_dicts: list[dict]) -> dict:
    """Share of predicted labels per sentiment group (counts of predicted labels)."""
    counts = {g: 0 for g in EMOTION_GROUPS}
    for preds in emo_dicts:
        for lab in preds:
            counts[LABEL_TO_GROUP.get(lab, "ambiguous")] += 1
    total = sum(counts.values()) or 1
    return {g: round(c / total, 4) for g, c in counts.items()}


def _top_emotions(df: pd.DataFrame, k: int = 3, exclude_neutral: bool = True) -> list[dict]:
    counter: dict[str, int] = {}
    for preds in df["predicted"]:
        for lab in preds:
            if exclude_neutral and lab == "neutral":
                continue
            counter[lab] = counter.get(lab, 0) + 1
    return [{"label": lab, "entries": n} for lab, n in sorted(counter.items(), key=lambda x: -x[1])[:k]]


def _aggregate(g: pd.DataFrame) -> dict:
    emo_mean: dict[str, float] = {}
    if len(g):
        keys = set().union(*[d.keys() for d in g["emo"]])
        emo_mean = {k: round(float(np.mean([d.get(k, 0.0) for d in g["emo"]])), 4) for k in keys}
    return {
        "entries": int(len(g)),
        "avg_stress_score": _mean(g["stress_score"]),
        "stress_language_share": _mean(g["stress_flag"]),
        "avg_mood": _mean(g["mood"]), "avg_energy": _mean(g["energy"]),
        "avg_sleep_hours": _mean(g["sleep_hours"]), "avg_sleep_quality": _mean(g["sleep_quality"]),
        "emotion_groups": _group_scores(list(g["predicted"])),
        "top_emotions": _top_emotions(g),
        "mean_emotion_scores": emo_mean,
    }


def daily_series(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    return [{"date": d.strftime("%Y-%m-%d"), **_aggregate(g)} for d, g in df.groupby("date")]


def period_summaries(df: pd.DataFrame, freq: str) -> list[dict]:
    """freq: 'W' (weeks starting Monday) or 'M' (calendar months)."""
    if df.empty:
        return []
    key = df["created_at"].dt.to_period("W-SUN" if freq == "W" else "M")
    out, prev = [], None
    for period, g in df.groupby(key):
        agg = _aggregate(g)
        item = {"period": str(period), "start": period.start_time.strftime("%Y-%m-%d"),
                "end": period.end_time.strftime("%Y-%m-%d"),
                "label": (f"Week of {period.start_time:%d %b %Y}" if freq == "W" else f"{period.start_time:%B %Y}"),
                **agg}
        item["observations"] = _compare(prev, agg, "week" if freq == "W" else "month") if prev else []
        out.append(item)
        prev = agg
    return out


def _compare(prev: dict, cur: dict, unit: str, min_entries: int = 2) -> list[str]:
    obs = []
    if prev["entries"] < min_entries or cur["entries"] < min_entries:
        return [f"Too few entries to compare with the previous {unit} (at least {min_entries} in each needed)."]
    if prev["avg_stress_score"] is not None and cur["avg_stress_score"] is not None:
        d = cur["avg_stress_score"] - prev["avg_stress_score"]
        if d >= 0.1:
            obs.append(f"Stress-related language increased compared with the previous {unit}.")
        elif d <= -0.1:
            obs.append(f"Stress-related language decreased compared with the previous {unit}.")
        else:
            obs.append(f"Stress-related language was broadly similar to the previous {unit}.")
    for grp in ("positive", "negative"):
        d = cur["emotion_groups"][grp] - prev["emotion_groups"][grp]
        if abs(d) >= 0.15:
            obs.append(f"A {'larger' if d > 0 else 'smaller'} share of detected emotions were in the {grp} group "
                       f"than in the previous {unit}.")
    if prev["avg_mood"] is not None and cur["avg_mood"] is not None and abs(cur["avg_mood"] - prev["avg_mood"]) >= 0.75:
        obs.append(f"Your self-reported mood check-ins were {'higher' if cur['avg_mood'] > prev['avg_mood'] else 'lower'} "
                   f"on average than the previous {unit}.")
    return obs


def recent_observations(df: pd.DataFrame, window_days: int = 7) -> dict:
    """Compare the most recent window with the one before it."""
    if df.empty:
        return {"statements": [], "window_days": window_days}
    end = df["created_at"].max()
    recent = df[df["created_at"] > end - timedelta(days=window_days)]
    before = df[(df["created_at"] <= end - timedelta(days=window_days)) &
                (df["created_at"] > end - timedelta(days=2 * window_days))]
    statements = []
    if len(recent) >= 2 and len(before) >= 2:
        r, b = _aggregate(recent), _aggregate(before)
        d = (r["avg_stress_score"] or 0) - (b["avg_stress_score"] or 0)
        if d >= 0.1:
            statements.append("Stress-related language has increased across your recent entries.")
        elif d <= -0.1:
            statements.append("Stress-related language has decreased across your recent entries.")
        else:
            statements.append("Stress-related language in your recent entries is similar to the week before.")
        top_now = [t["label"] for t in r["top_emotions"]]
        top_before = {t["label"] for t in b["top_emotions"]}
        new = [t for t in top_now if t not in top_before]
        if new:
            statements.append(f"Language associated with {', '.join(new)} appeared more prominently in recent entries.")
    elif len(df) >= 1:
        statements.append(f"Add entries over at least two {window_days}-day periods to see how your language "
                          "patterns change over time.")
    # Self-report vs. language relationship (descriptive correlation only).
    sub = df.dropna(subset=["sleep_hours", "stress_score"])
    if len(sub) >= 6 and sub["sleep_hours"].std() > 0:
        rho = float(sub["sleep_hours"].corr(sub["stress_score"], method="spearman"))
        if abs(rho) >= 0.3:
            statements.append(
                f"In your entries, nights with {'less' if rho < 0 else 'more'} reported sleep tended to coincide with "
                f"more stress-related language (Spearman rho = {rho:.2f}, n = {len(sub)}). This is a correlation "
                "in your own data, not evidence of cause.")
    return {"statements": statements, "window_days": window_days}


def emotion_distribution(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    labels = set().union(*[d.keys() for d in df["emo"]])
    rows = []
    for lab in labels:
        rows.append({"label": lab, "group": LABEL_TO_GROUP.get(lab, "ambiguous"),
                     "entries_predicted": int(sum(lab in p for p in df["predicted"])),
                     "mean_score": round(float(np.mean([d.get(lab, 0.0) for d in df["emo"]])), 4)})
    return sorted(rows, key=lambda r: (-r["entries_predicted"], -r["mean_score"]))


def filter_days(df: pd.DataFrame, days: int | None) -> pd.DataFrame:
    if df.empty or not days:
        return df
    return df[df["created_at"] >= datetime.utcnow() - timedelta(days=days)]
