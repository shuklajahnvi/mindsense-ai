"""Application settings, read from environment variables / backend/.env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(BACKEND_DIR / ".env", BACKEND_DIR.parent / ".env"),
                                      env_file_encoding="utf-8", extra="ignore")

    database_url: str = f"sqlite:///{(DATA_DIR / 'mindsense.db').as_posix()}"
    cors_origins: str = "http://localhost:3004,http://127.0.0.1:3004"

    # Model selection: "auto" uses the fine-tuned transformer when it exists, else the baseline.
    emotion_model: str = "auto"   # auto | transformer | baseline
    stress_model: str = "auto"    # auto | transformer | baseline
    lime_num_samples: int = 300
    lime_num_features: int = 10

    # Sentence embeddings (RAG + topic analysis)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = str(DATA_DIR / "chroma")
    knowledge_base_dir: str = str(BACKEND_DIR / "knowledge_base")
    rag_top_k: int = 4
    rag_min_similarity: float = 0.2

    # Generative AI provider - keys are only ever read from the environment.
    llm_provider: str = "auto"    # auto | openai | gemini | none
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    llm_timeout_seconds: float = 45.0
    llm_temperature: float = 0.4


@lru_cache
def get_settings() -> Settings:
    return Settings()
