"""LIME explanations for individual emotion and stress predictions.

LIME (Ribeiro et al., 2016) perturbs the input by removing words, queries the
model on the perturbed texts and fits a sparse local linear model. The
resulting weights describe which words pushed *this model's* score for *this
label* up (positive) or down (negative) for this one text. They describe model
behaviour; they say nothing about causes of a person's feelings.

LIME is model-agnostic (only needs ``predict_proba``) so it works for both the
TF-IDF baselines and the fine-tuned transformers. For multi-label emotion we
explain each label's independent sigmoid probability; LIME's per-label ridge
regression does not require probabilities to sum to one.
"""
from __future__ import annotations

import time

from lime.lime_text import LimeTextExplainer

from app.config import get_settings
from app.services.classifiers import registry

# Split on runs of non-word characters except apostrophes, so contractions ("can't", "I'm") stay whole.
# The frontend highlighter uses the identical regex.
SPLIT_EXPRESSION = r"[^\w']+"

DISCLAIMER = ("These words contributed to the model's prediction for this text. They show how the model "
              "reacted to the wording; they do not establish your psychological state or what caused your feelings.")


def _explain(model, text: str, label_indices: list[int], num_samples: int, num_features: int) -> dict:
    explainer = LimeTextExplainer(class_names=model.labels, bow=True, random_state=42,
                                  split_expression=SPLIT_EXPRESSION)
    t0 = time.time()
    exp = explainer.explain_instance(text, model.predict_proba, labels=label_indices,
                                     num_features=num_features, num_samples=num_samples)
    out = []
    for idx in label_indices:
        weights = [{"token": tok, "weight": float(w)} for tok, w in exp.as_list(label=idx)]
        out.append({
            "label": model.labels[idx],
            "prediction_score": float(exp.predict_proba[idx]) if exp.predict_proba is not None else None,
            "local_model_r2": float(exp.score[idx]) if isinstance(exp.score, dict) else float(exp.score),
            "intercept": float(exp.intercept[idx]),
            "weights": weights,
        })
    return {"targets": out, "seconds": round(time.time() - t0, 2)}


def explain_entry(text: str, emotion_rows: list[dict] | None, emotion_variant: str | None = None,
                  stress_variant: str | None = None, num_samples: int | None = None, top_emotions: int = 2) -> dict:
    s = get_settings()
    n = num_samples or s.lime_num_samples
    result: dict = {"method": "LIME (LimeTextExplainer, bag-of-words perturbation)", "num_samples": n,
                    "split_expression": SPLIT_EXPRESSION,
                    "disclaimer": DISCLAIMER}
    em = registry.get("emotion", emotion_variant)
    if em is not None and emotion_rows:
        # Explain the highest-scoring predicted emotions (fall back to the top score if none crosses threshold).
        chosen = [r["label"] for r in emotion_rows if r["predicted"]][:top_emotions] or [emotion_rows[0]["label"]]
        idx = [em.labels.index(lbl) for lbl in chosen]
        result["emotion"] = {"model": em.info()["name"], "variant": em.variant,
                             **_explain(em, text, idx, n, s.lime_num_features)}
    st = registry.get("stress", stress_variant)
    if st is not None:
        result["stress"] = {"model": st.info()["name"], "variant": st.variant,
                            **_explain(st, text, [1], n, s.lime_num_features)}
    return result
