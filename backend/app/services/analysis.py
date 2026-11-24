"""Run the full per-entry analysis: safety screen, emotion + stress models, LIME."""
from __future__ import annotations

from app.services import explain, safety
from app.services.classifiers import predict_emotions, predict_stress


class ModelsUnavailable(RuntimeError):
    pass


def analyze_text(text: str, do_explain: bool = True, emotion_model: str = "auto", stress_model: str = "auto",
                 lime_samples: int | None = None) -> dict:
    ev = None if emotion_model == "auto" else emotion_model
    sv = None if stress_model == "auto" else stress_model
    emo = predict_emotions(text, ev)
    st = predict_stress(text, sv)
    if emo is None and st is None:
        raise ModelsUnavailable("No trained models found. Run the training scripts in backend/ml first "
                                "(see README).")
    result: dict = {"safety": safety.assess_text(text), "model_info": {}, "notice": safety.NON_CLINICAL_NOTICE}
    if emo:
        rows, info = emo
        result["emotions"] = rows
        result["predicted_emotions"] = [r["label"] for r in rows if r["predicted"]]
        result["model_info"]["emotion"] = info
    else:
        result.update(emotions=None, predicted_emotions=None)
        result["model_info"]["emotion"] = {"status": "unavailable"}
    if st:
        pred, info = st
        result["stress"] = pred
        result["model_info"]["stress"] = info
    else:
        result["stress"] = None
        result["model_info"]["stress"] = {"status": "unavailable"}
    result["explanations"] = (explain.explain_entry(text, result.get("emotions"), ev, sv, lime_samples)
                              if do_explain else None)
    return result
