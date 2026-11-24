"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Variant = Literal["auto", "transformer", "baseline"]


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=3, max_length=10_000)
    explain: bool = True
    emotion_model: Variant = "auto"
    stress_model: Variant = "auto"
    lime_samples: int | None = Field(default=None, ge=50, le=2000)


class EntryCreate(AnalyzeRequest):
    mood: int | None = Field(default=None, ge=1, le=5)
    energy: int | None = Field(default=None, ge=1, le=5)
    sleep_hours: float | None = Field(default=None, ge=0, le=24)
    sleep_quality: int | None = Field(default=None, ge=1, le=5)
    created_at: datetime | None = Field(default=None, description="Optional back-dating of an entry")


class ExplainRequest(BaseModel):
    emotion_model: Variant = "auto"
    stress_model: Variant = "auto"
    lime_samples: int | None = Field(default=None, ge=50, le=2000)


class EntryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: int
    created_at: datetime
    text: str
    mood: int | None
    energy: int | None
    sleep_hours: float | None
    sleep_quality: int | None
    emotions: list[dict] | None
    predicted_emotions: list[str] | None
    stress_label: str | None
    stress_score: float | None
    model_info: dict | None
    explanations: dict | None
    safety: dict | None
    reflection: dict | None
    is_demo: bool


class EntrySummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
    text: str
    mood: int | None
    energy: int | None
    sleep_hours: float | None
    predicted_emotions: list[str] | None
    stress_label: str | None
    stress_score: float | None
    is_demo: bool


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    use_journal_context: bool = True


class ChatResponse(BaseModel):
    answer: str
    mode: Literal["llm", "retrieval_only", "safety"]
    provider: dict | None
    sources: list[dict]
    safety: dict
    guardrail_removed: list[str]
    error: str | None
    message_id: int | None = None


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
    role: str
    content: str
    sources: list[Any] | None
    provider: str | None
    safety: dict | None
