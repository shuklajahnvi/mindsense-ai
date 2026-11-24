from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.api.insights import personal_context
from app.db import get_db
from app.models import ChatMessage
from app.schemas import ChatMessageOut, ChatRequest, ChatResponse
from app.services import assistant

router = APIRouter(prefix="/api/chat", tags=["assistant"])
USER_ID = 1


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    history = db.scalars(select(ChatMessage).where(ChatMessage.user_id == USER_ID)
                         .order_by(ChatMessage.id.desc()).limit(6)).all()[::-1]
    hist = [{"role": m.role, "content": m.content} for m in history if m.role in ("user", "assistant")]
    ctx = personal_context(db) if req.use_journal_context else None
    res = assistant.chat(req.message, hist, ctx)
    db.add(ChatMessage(user_id=USER_ID, role="user", content=req.message, safety=res["safety"]))
    reply = ChatMessage(user_id=USER_ID, role="assistant", content=res["answer"], sources=res["sources"],
                        provider=(res["provider"] or {}).get("provider") if res["mode"] == "llm" else res["mode"],
                        safety=res["safety"])
    db.add(reply); db.commit()
    return {**res, "message_id": reply.id}


@router.get("/history", response_model=list[ChatMessageOut])
def history(db: Session = Depends(get_db)):
    return db.scalars(select(ChatMessage).where(ChatMessage.user_id == USER_ID).order_by(ChatMessage.id)).all()


@router.delete("/history")
def clear_history(db: Session = Depends(get_db)):
    db.execute(delete(ChatMessage).where(ChatMessage.user_id == USER_ID)); db.commit()
    return {"cleared": True}
