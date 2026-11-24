from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import JournalEntry
from app.schemas import AnalyzeRequest, EntryCreate, EntryOut, EntrySummary, ExplainRequest
from app.services import assistant, explain
from app.services.analysis import ModelsUnavailable, analyze_text

router = APIRouter(prefix="/api", tags=["entries"])
USER_ID = 1  # single local demo profile


def _apply(entry: JournalEntry, res: dict) -> None:
    entry.emotions = res.get("emotions")
    entry.predicted_emotions = res.get("predicted_emotions")
    entry.stress_label = res["stress"]["label"] if res.get("stress") else None
    entry.stress_score = res["stress"]["score"] if res.get("stress") else None
    entry.model_info = res["model_info"]
    entry.explanations = res.get("explanations")
    entry.safety = res["safety"]


def _get(db: Session, entry_id: int) -> JournalEntry:
    e = db.get(JournalEntry, entry_id)
    if e is None or e.user_id != USER_ID:
        raise HTTPException(404, "Entry not found")
    return e


@router.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Analyse text without saving it."""
    try:
        return analyze_text(req.text, req.explain, req.emotion_model, req.stress_model, req.lime_samples)
    except ModelsUnavailable as e:
        raise HTTPException(503, str(e))


@router.post("/entries", response_model=EntryOut)
def create_entry(req: EntryCreate, db: Session = Depends(get_db)):
    try:
        res = analyze_text(req.text, req.explain, req.emotion_model, req.stress_model, req.lime_samples)
    except ModelsUnavailable as e:
        raise HTTPException(503, str(e))
    entry = JournalEntry(user_id=USER_ID, text=req.text, mood=req.mood, energy=req.energy,
                         sleep_hours=req.sleep_hours, sleep_quality=req.sleep_quality)
    if req.created_at:
        entry.created_at = req.created_at.replace(tzinfo=None)
    _apply(entry, res)
    db.add(entry); db.commit(); db.refresh(entry)
    return entry


@router.get("/entries")
def list_entries(limit: int = Query(50, le=500), offset: int = 0, db: Session = Depends(get_db)):
    q = select(JournalEntry).where(JournalEntry.user_id == USER_ID)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(q.order_by(JournalEntry.created_at.desc()).limit(limit).offset(offset)).all()
    return {"total": total, "items": [EntrySummary.model_validate(r) for r in rows]}


@router.get("/entries/{entry_id}", response_model=EntryOut)
def get_entry(entry_id: int, db: Session = Depends(get_db)):
    return _get(db, entry_id)


@router.delete("/entries/{entry_id}")
def delete_entry(entry_id: int, db: Session = Depends(get_db)):
    db.delete(_get(db, entry_id)); db.commit()
    return {"deleted": entry_id}


@router.post("/entries/{entry_id}/explain", response_model=EntryOut)
def re_explain(entry_id: int, req: ExplainRequest, db: Session = Depends(get_db)):
    """(Re)compute LIME explanations, optionally for a different model variant."""
    e = _get(db, entry_id)
    try:
        res = analyze_text(e.text, False, req.emotion_model, req.stress_model)
    except ModelsUnavailable as ex:
        raise HTTPException(503, str(ex))
    ev = None if req.emotion_model == "auto" else req.emotion_model
    sv = None if req.stress_model == "auto" else req.stress_model
    if req.emotion_model == "auto" and req.stress_model == "auto":
        _apply(e, res)  # refresh stored predictions only for the default models
        e.explanations = explain.explain_entry(e.text, e.emotions, None, None, req.lime_samples)
    else:
        e.explanations = explain.explain_entry(e.text, res.get("emotions"), ev, sv, req.lime_samples)
    db.commit(); db.refresh(e)
    return e


@router.post("/entries/{entry_id}/reflection", response_model=EntryOut)
def reflection(entry_id: int, db: Session = Depends(get_db)):
    e = _get(db, entry_id)
    e.reflection = assistant.reflect_on_entry(e)
    db.commit(); db.refresh(e)
    return e
