"""Shared sentence-embedding model (Sentence Transformers), loaded lazily once."""
from __future__ import annotations

import threading

import numpy as np

from app.config import get_settings

_model = None
_lock = threading.Lock()


def get_embedder():
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(get_settings().embedding_model, device="cpu")
        return _model


def embed(texts: list[str]) -> np.ndarray:
    return np.asarray(get_embedder().encode(texts, normalize_embeddings=True, batch_size=32,
                                            show_progress_bar=False))
