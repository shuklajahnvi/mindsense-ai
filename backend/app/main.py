"""MindSense AI - FastAPI application entry point.

Run from the backend/ directory:
    .venv\\Scripts\\python -m uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import chat, demo, entries, insights, system
from app.config import get_settings
from app.db import init_db
from ml.paths import REPORTS_DIR

log = logging.getLogger("mindsense")


def _warmup() -> None:
    """Build the RAG index and load models in the background so the first request is fast."""
    try:
        from app.services import rag
        from app.services.classifiers import registry

        rag.build_index()
        for task in ("emotion", "stress"):
            registry.get(task)
        log.info("MindSense warm-up complete")
    except Exception:  # pragma: no cover
        log.exception("Warm-up failed (the app still runs; components load lazily)")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(title="MindSense AI API", version="1.0.0", lifespan=lifespan,
              description="Non-clinical journaling analysis: emotion & stress-language classification, LIME "
                          "explanations, longitudinal trends, and a RAG-grounded reflective assistant.")
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins.split(","), allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
for r in (system.router, entries.router, insights.router, chat.router, demo.router):
    app.include_router(r)
app.mount("/api/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")
