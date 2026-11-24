from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.api.entries import USER_ID, _apply
from app.db import get_db
from app.demo_data import DEMO_ENTRIES
from app.models import JournalEntry
from app.services.analysis import ModelsUnavailable, analyze_text

router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.post("/seed")
def seed(db: Session = Depends(get_db)):
    """Insert synthetic demo entries, each analysed by the real models (LIME computed on demand)."""
    db.execute(delete(JournalEntry).where(JournalEntry.is_demo.is_(True))); db.commit()
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    n = 0
    for days_ago, hour, mood, energy, sleep, sq, text in DEMO_ENTRIES:
        try:
            res = analyze_text(text, do_explain=False)
        except ModelsUnavailable as e:
            raise HTTPException(503, str(e))
        e = JournalEntry(user_id=USER_ID, text=text, mood=mood, energy=energy, sleep_hours=sleep,
                         sleep_quality=sq, is_demo=True,
                         created_at=(now - timedelta(days=days_ago)).replace(hour=hour % 24))
        _apply(e, res)
        db.add(e); n += 1
    db.commit()
    return {"inserted": n, "note": "Synthetic demo text; predictions are real model outputs."}


@router.delete("")
def clear_demo(db: Session = Depends(get_db)):
    r = db.execute(delete(JournalEntry).where(JournalEntry.is_demo.is_(True))); db.commit()
    return {"deleted": r.rowcount}
