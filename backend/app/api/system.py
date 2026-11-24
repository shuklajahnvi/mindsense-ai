from __future__ import annotations

import json

from fastapi import APIRouter

from app.services import llm, rag, safety
from app.services.classifiers import registry
from ml.paths import REPORTS_DIR

router = APIRouter(prefix="/api", tags=["system"])


def _summarise_meta(meta: dict | None) -> dict | None:
    if not meta:
        return None
    return {k: meta.get(k) for k in ("name", "task", "kind", "algorithm", "fine_tuned", "base_checkpoint",
                                     "fine_tuning_scope", "trainable_parameters", "total_parameters",
                                     "labels", "hyperparameters", "training", "metrics", "trained_at")}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/status")
def status():
    models = {}
    for task in ("emotion", "stress"):
        avail = registry.available(task)
        models[task] = {
            "active_variant": registry.resolve_variant(task),
            "available": {v: (None if m is None else {"name": m["name"], "fine_tuned": m.get("fine_tuned", False),
                                                      "algorithm": m["algorithm"],
                                                      "trained_on_subset": m.get("training", {}).get("is_subset", False)})
                          for v, m in avail.items()},
        }
    return {"models": models, "llm": llm.provider_status(), "rag": rag.index_status(),
            "explainability": {"method": "LIME", "available": any(m["active_variant"] for m in models.values())},
            "notice": safety.NON_CLINICAL_NOTICE}


@router.get("/models")
def model_reports():
    """Full metadata + evaluation metrics for every trained model, plus EDA summaries."""
    out = {"emotion": {}, "stress": {}, "eda": {}}
    for task in ("emotion", "stress"):
        for f in sorted((REPORTS_DIR / task).glob("*_metrics.json")):
            meta = json.loads(f.read_text())
            out[task][meta["name"]] = _summarise_meta(meta)
        eda = REPORTS_DIR / task / "eda.json"
        out["eda"][task] = json.loads(eda.read_text()) if eda.exists() else None
        out["eda"][f"{task}_figures"] = sorted(p.name for p in (REPORTS_DIR / task).glob("*.png"))
    return out


@router.post("/models/reload")
def reload_models():
    registry.reload()
    return status()


@router.get("/safety/resources")
def resources():
    return {"notice": safety.NON_CLINICAL_NOTICE, "resources": safety.SUPPORT_RESOURCES}
