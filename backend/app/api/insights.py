from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import JournalEntry
from app.services import topics, trends

router = APIRouter(prefix="/api/insights", tags=["insights"])
USER_ID = 1


def _entries(db: Session):
    return db.scalars(select(JournalEntry).where(JournalEntry.user_id == USER_ID)
                      .order_by(JournalEntry.created_at)).all()


@router.get("")
def insights(days: int | None = Query(None, ge=1, le=3650), db: Session = Depends(get_db)):
    df = trends.filter_days(trends.to_frame(_entries(db)), days)
    return {
        "n_entries": int(len(df)),
        "caveat": trends.CAVEAT,
        "observations": trends.recent_observations(df),
        "daily": trends.daily_series(df),
        "weekly": trends.period_summaries(df, "W"),
        "monthly": trends.period_summaries(df, "M"),
        "emotion_distribution": trends.emotion_distribution(df),
        "emotion_groups": trends.EMOTION_GROUPS,
    }


@router.get("/topics")
def recurring_topics(db: Session = Depends(get_db)):
    return topics.find_topics(db, list(_entries(db)))


def personal_context(db: Session, days: int = 14) -> str | None:
    """Short, non-clinical description of recent language patterns for the assistant."""
    df = trends.filter_days(trends.to_frame(_entries(db)), days)
    if df.empty:
        return None
    agg = trends._aggregate(df)
    parts = [f"{agg['entries']} entries in the last {days} days."]
    if agg["top_emotions"]:
        parts.append("Most frequent emotion labels in the language: " +
                     ", ".join(t["label"] for t in agg["top_emotions"]) + ".")
    if agg["stress_language_share"] is not None:
        parts.append(f"Stress-related language was detected in {agg['stress_language_share'] * 100:.0f}% of entries.")
    parts += trends.recent_observations(df)["statements"][:2]
    return " ".join(parts)
