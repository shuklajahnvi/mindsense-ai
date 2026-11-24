"""Fast unit tests that don't need trained models or network access."""
import numpy as np

from app.services import safety
from app.services.rag import _parse_doc, chunk_document
from ml.common.metrics import binary_report, multilabel_report, tune_thresholds
from ml.common.text import clean_text


def test_clean_text():
    assert clean_text("see https://x.com &amp; u/someone  now") == "see [URL] & [USER] now"


def test_safety_high_risk():
    r = safety.assess_text("I want to kill myself tonight")
    assert r["level"] == "high" and "suicide_or_self_harm" in r["categories"] and r["resources"]


def test_safety_negation_not_flagged():
    assert safety.assess_text("I would never hurt myself, I just feel tired").get("level") == "none"


def test_safety_elevated():
    assert safety.assess_text("Everything feels hopeless lately").get("level") == "elevated"


def test_safety_none():
    r = safety.assess_text("Had a nice walk and coffee with a friend.")
    assert r["level"] == "none" and r["resources"] == []


def test_sanitize_removes_diagnosis_and_medication():
    text = ("It sounds like a tough week. You may have depression. Try taking 50 mg of sertraline.\n"
            "- Take a short walk [1]")
    out, removed = safety.sanitize_generated(text)
    assert "depression" not in out and "sertraline" not in out
    assert "tough week" in out and "- Take a short walk [1]" in out
    assert set(removed) == {"diagnostic_claim", "medication_advice"}


def test_sanitize_keeps_clean_text():
    text = "### Ideas\n- Keep a regular sleep schedule [1]\n- Talk to someone you trust"
    out, removed = safety.sanitize_generated(text)
    assert out == text and removed == []


def test_chunking_and_front_matter(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("---\ntitle: T\ntopic: x\nsources:\n  - name: A\n    url: https://a\n---\n\n## One\n\n"
                 + ("word " * 400) + "\n\n## Two\n\nshort para", encoding="utf-8")
    meta, body = _parse_doc(f)
    assert meta["title"] == "T" and meta["sources"][0]["url"] == "https://a"
    chunks = chunk_document(body, max_words=100, overlap_words=10)
    assert {h for h, _ in chunks} == {"One", "Two"}
    assert all(len(t.split()) <= 150 for _, t in chunks)


def test_multilabel_metrics_and_thresholds():
    y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6], [0.1, 0.2]])
    th = tune_thresholds(y, p)
    rep = multilabel_report(y, p, th, ["a", "b"])
    assert rep["micro_f1"] == 1.0 and rep["macro_f1"] == 1.0


def test_binary_metrics():
    rep = binary_report(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.4, 0.3]))
    assert rep["accuracy"] == 0.75 and rep["confusion_matrix"]["matrix"] == [[2, 0], [1, 1]]
