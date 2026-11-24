"""API integration tests. Model-dependent tests are skipped when models aren't trained."""
import os

import pytest

os.environ["DATABASE_URL"] = "sqlite:///./data/test_mindsense.db"
os.environ["LLM_PROVIDER"] = "none"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.classifiers import registry  # noqa: E402

HAS_MODELS = registry.resolve_variant("emotion") is not None and registry.resolve_variant("stress") is not None


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_status(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    s = client.get("/api/status").json()
    assert {"models", "llm", "rag", "notice"} <= s.keys()
    assert s["llm"]["configured"] is False


def test_chat_retrieval_only(client):
    r = client.post("/api/chat", json={"message": "How can I sleep better when stressed?"}).json()
    assert r["mode"] == "retrieval_only" and r["sources"], r
    assert any("sleep" in s["title"].lower() for s in r["sources"])


def test_chat_safety_bypass(client):
    r = client.post("/api/chat", json={"message": "I want to end my life"}).json()
    assert r["mode"] == "safety" and r["safety"]["level"] == "high" and r["safety"]["resources"]


@pytest.mark.skipif(not HAS_MODELS, reason="models not trained")
def test_entry_lifecycle(client):
    body = {"text": "I'm so stressed about my exams, I can't sleep and I'm worried I'll fail.",
            "mood": 2, "sleep_hours": 5, "explain": True, "lime_samples": 100}
    e = client.post("/api/entries", json=body).json()
    assert len(e["emotions"]) == 28 and 0 <= e["stress_score"] <= 1
    assert e["explanations"]["stress"]["targets"][0]["weights"]
    got = client.get(f"/api/entries/{e['id']}").json()
    assert got["text"] == body["text"]
    refl = client.post(f"/api/entries/{e['id']}/reflection").json()["reflection"]
    assert refl["mode"] == "retrieval_only" and "reflect" in refl["answer"].lower()
    assert client.get("/api/insights").json()["n_entries"] >= 1
    assert client.delete(f"/api/entries/{e['id']}").status_code == 200
