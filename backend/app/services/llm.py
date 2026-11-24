"""Provider-independent LLM interface.

Providers are selected with ``LLM_PROVIDER`` (auto | openai | gemini | none) and
API keys are read only from the environment. ``auto`` picks OpenAI if
``OPENAI_API_KEY`` is set, else Gemini if ``GEMINI_API_KEY`` is set, else none.
When no provider is configured, callers fall back to a transparent
retrieval-only response (see ``app.services.assistant``) - nothing is presented
as LLM output unless an LLM actually produced it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from app.config import get_settings


@dataclass
class ChatTurn:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMError(RuntimeError):
    pass


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def generate(self, messages: list[ChatTurn], temperature: float, max_tokens: int = 700) -> str: ...

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model, "configured": True}


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float):
        self.api_key, self.model, self.base_url, self.timeout = api_key, model, base_url.rstrip("/"), timeout

    def generate(self, messages, temperature, max_tokens=700) -> str:
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", timeout=self.timeout,
                           headers={"Authorization": f"Bearer {self.api_key}"},
                           json={"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                                 "messages": [{"role": m.role, "content": m.content} for m in messages]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            raise LLMError(f"OpenAI API error {e.response.status_code}: {e.response.text[:300]}") from e
        except (httpx.HTTPError, KeyError, IndexError) as e:
            raise LLMError(f"OpenAI request failed: {e}") from e


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout: float):
        self.api_key, self.model, self.timeout = api_key, model, timeout

    def generate(self, messages, temperature, max_tokens=700) -> str:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [{"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
                    for m in messages if m.role != "system"]
        body = {"contents": contents,
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        try:
            r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                           headers={"x-goog-api-key": self.api_key}, json=body, timeout=self.timeout)
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Gemini API error {e.response.status_code}: {e.response.text[:300]}") from e
        except (httpx.HTTPError, KeyError, IndexError) as e:
            raise LLMError(f"Gemini request failed: {e}") from e


def get_provider() -> LLMProvider | None:
    s = get_settings()
    choice = (s.llm_provider or "auto").lower()
    if choice in ("auto", "openai") and s.openai_api_key:
        return OpenAIProvider(s.openai_api_key, s.openai_model, s.openai_base_url, s.llm_timeout_seconds)
    if choice in ("auto", "gemini") and s.gemini_api_key:
        return GeminiProvider(s.gemini_api_key, s.gemini_model, s.llm_timeout_seconds)
    return None


def provider_status() -> dict:
    p = get_provider()
    if p:
        return p.describe()
    s = get_settings()
    return {"provider": "none", "model": None, "configured": False,
            "detail": "No LLM API key configured (set OPENAI_API_KEY or GEMINI_API_KEY in backend/.env). "
                      "The assistant runs in retrieval-only mode." if s.llm_provider != "none"
            else "LLM disabled by LLM_PROVIDER=none."}
