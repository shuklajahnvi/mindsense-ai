"""SQLAlchemy ORM tables."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    display_name: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class JournalEntry(Base):
    __tablename__ = "journal_entries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    text: Mapped[str] = mapped_column(Text)

    # Optional self-reported check-ins
    mood: Mapped[int | None] = mapped_column(Integer, nullable=True)          # 1 (very low) .. 5 (very good)
    energy: Mapped[int | None] = mapped_column(Integer, nullable=True)        # 1 .. 5
    sleep_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    sleep_quality: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1 .. 5

    # Model outputs
    emotions: Mapped[list | None] = mapped_column(JSON, nullable=True)        # [{label, score, threshold, predicted}]
    predicted_emotions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    stress_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    stress_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    explanations: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    safety: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    reflection: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    safety: Mapped[dict | None] = mapped_column(JSON, nullable=True)
