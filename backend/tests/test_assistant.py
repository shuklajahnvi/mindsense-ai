"""Generation path of the RAG assistant, using a fake provider (no network / API key needed)."""
from app.services import assistant
from app.services.llm import LLMError, LLMProvider


class FakeProvider(LLMProvider):
    name, model = "fake", "fake-1"

    def __init__(self, reply=None, fail=False):
        self.reply, self.fail, self.seen = reply, fail, None

    def generate(self, messages, temperature, max_tokens=700):
        self.seen = messages
        if self.fail:
            raise LLMError("boom")
        return self.reply


def test_llm_answer_is_grounded_and_guarded(monkeypatch):
    fake = FakeProvider("Try keeping a regular bedtime [1]. You clearly have an anxiety disorder.")
    monkeypatch.setattr(assistant, "get_provider", lambda: fake)
    r = assistant.chat("How can I sleep better?", personal_context="3 entries in the last 14 days.")
    assert r["mode"] == "llm" and r["provider"]["provider"] == "fake"
    assert "anxiety disorder" not in r["answer"] and "regular bedtime [1]" in r["answer"]
    assert r["guardrail_removed"] == ["diagnostic_claim"]
    prompt = fake.seen[-1].content
    assert "CONTEXT:" in prompt and "[1]" in prompt and "3 entries in the last 14 days." in prompt
    assert fake.seen[0].role == "system" and "Never diagnose" in fake.seen[0].content


def test_llm_failure_falls_back_to_retrieval(monkeypatch):
    monkeypatch.setattr(assistant, "get_provider", lambda: FakeProvider(fail=True))
    r = assistant.chat("Tips for exam stress?")
    assert r["mode"] == "retrieval_only" and r["error"] == "boom" and r["sources"]


def test_high_risk_message_never_reaches_llm(monkeypatch):
    fake = FakeProvider("should not be used")
    monkeypatch.setattr(assistant, "get_provider", lambda: fake)
    r = assistant.chat("I have been thinking about suicide")
    assert r["mode"] == "safety" and fake.seen is None
