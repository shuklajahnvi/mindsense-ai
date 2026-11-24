"""RAG assistant: grounded chat and personalised reflections on journal entries.

Flow for every request:
  safety screen on user text -> (if high risk: fixed supportive message + crisis
  resources, no generation) -> retrieve knowledge-base chunks -> build prompt
  with numbered context (+ optional non-clinical summary of the user's recent
  language patterns) -> LLM -> output guardrail (strip diagnostic / medication
  content). Without a configured LLM the response is a clearly labelled
  retrieval-only answer built from the retrieved passages.
"""
from __future__ import annotations

from app.config import get_settings
from app.services import rag, safety
from app.services.llm import ChatTurn, LLMError, get_provider

SYSTEM_PROMPT = """You are the reflective wellbeing assistant inside MindSense AI, a non-clinical journaling app.
You are not a therapist, doctor or crisis service.

Rules you must always follow:
- Never diagnose, label or suggest that the user has any mental health condition (e.g. depression, anxiety disorder, PTSD).
- Never recommend, adjust or discuss medication or dosages; suggest speaking to a qualified professional instead.
- Base practical suggestions on the numbered CONTEXT passages and cite them inline like [1] or [2]. If the context does not cover the question, say so briefly and keep any advice general.
- Emotion and stress scores from MindSense describe patterns in the *language* of entries, not the person's mental state. Refer to them that way (e.g. "your recent entries used more stress-related language").
- Do not interpret changes in scores as clinical improvement or deterioration.
- Be warm, non-judgemental, concise (under 220 words) and use plain language. Invite reflection rather than asserting how the user feels.
- If difficulties sound persistent or severe, gently encourage talking to a trusted person or professional. If there is any sign of risk of harm, encourage contacting emergency services or a crisis line."""


def _context_block(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] {c['title']} - {c['section']}\n{c['text']}" for i, c in enumerate(chunks))


def _sources(chunks: list[dict]) -> list[dict]:
    return [{"ref": i + 1, "title": c["title"], "section": c["section"], "similarity": c["similarity"],
             "excerpt": c["text"][:400] + ("..." if len(c["text"]) > 400 else ""), "sources": c["sources"]}
            for i, c in enumerate(chunks)]


def _retrieval_only(chunks: list[dict], intro: str) -> str:
    if not chunks:
        return (intro + "\n\nI couldn't find a closely matching passage in the knowledge base for this question. "
                "Try asking about stress, sleep, routines, journaling, study stress or where to find support.")
    lines = [intro, ""]
    for i, c in enumerate(chunks[:3]):
        lines.append(f"**[{i + 1}] {c['title']} - {c['section']}**")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines).strip()


def _generate(messages: list[ChatTurn]) -> tuple[str | None, dict, str | None]:
    provider = get_provider()
    if provider is None:
        return None, {"provider": "none", "model": None}, None
    try:
        text = provider.generate(messages, temperature=get_settings().llm_temperature)
        return text, {"provider": provider.name, "model": provider.model}, None
    except LLMError as e:
        return None, {"provider": provider.name, "model": provider.model}, str(e)


def _safety_response(check: dict) -> dict:
    return {"answer": check["message"], "mode": "safety", "provider": None, "sources": [], "safety": check,
            "guardrail_removed": [], "error": None}


def _finalise(text: str | None, chunks, info, err, fallback_intro: str, check: dict) -> dict:
    if text is not None:
        clean, removed = safety.sanitize_generated(text)
        if removed:
            clean += ("\n\n_Part of the generated response was removed by MindSense's safety filter because it "
                      "contained diagnostic or medication content. Please speak to a qualified professional "
                      "about those topics._")
        return {"answer": clean, "mode": "llm", "provider": info, "sources": _sources(chunks), "safety": check,
                "guardrail_removed": removed, "error": None}
    intro = fallback_intro if err is None else (
        f"The generative model could not be reached ({info['provider']}), so here are the most relevant "
        "passages from the curated knowledge base instead:")
    return {"answer": _retrieval_only(chunks, intro), "mode": "retrieval_only", "provider": info,
            "sources": _sources(chunks), "safety": check, "guardrail_removed": [], "error": err}


def chat(message: str, history: list[dict] | None = None, personal_context: str | None = None) -> dict:
    check = safety.assess_text(message)
    if check["level"] == "high":
        return _safety_response(check)
    chunks = rag.retrieve(message)
    msgs = [ChatTurn("system", SYSTEM_PROMPT)]
    for h in (history or [])[-6:]:
        msgs.append(ChatTurn(h["role"], h["content"]))
    user = f"CONTEXT:\n{_context_block(chunks) or '(no relevant passages found)'}\n\n"
    if personal_context:
        user += f"USER'S RECENT JOURNAL PATTERNS (model-derived, non-clinical):\n{personal_context}\n\n"
    user += f"QUESTION:\n{message}"
    if check["level"] == "elevated":
        user += ("\n\n(Note: the user's message contains language that may indicate they are struggling. Respond with "
                 "extra warmth and mention that talking to someone they trust or a support line can help.)")
    msgs.append(ChatTurn("user", user))
    text, info, err = _generate(msgs)
    res = _finalise(text, chunks, info, err,
                    "Generative AI is not configured, so this is a retrieval-only answer: the most relevant "
                    "passages from MindSense's curated knowledge base are shown below.", check)
    if check["level"] == "elevated" and res["mode"] != "llm":
        res["answer"] = check["message"] + "\n\n" + res["answer"]
    return res


# Reflection prompts used in retrieval-only mode, keyed by GoEmotions label.
_PROMPTS = {
    "nervousness": "What specifically feels uncertain right now, and which part of it is within your control?",
    "fear": "What would help you feel even a little safer or more prepared?",
    "sadness": "What, or who, has helped you before when you felt this way?",
    "grief": "Is there a way to honour what you have lost that feels right to you?",
    "anger": "What boundary or value of yours feels like it was crossed?",
    "annoyance": "Is this frustration pointing to something you'd like to change?",
    "disappointment": "What did you hope would happen, and what does that tell you about what matters to you?",
    "remorse": "What would you say to a friend who felt this way about a similar situation?",
    "embarrassment": "How much will this matter in a week or a month?",
    "confusion": "What is one question that, if answered, would make things clearer?",
    "gratitude": "What made this moment of gratitude possible, and could you create more of it?",
    "joy": "What contributed to this good moment, and how could you notice it again?",
    "optimism": "What small step could you take to move toward what you're hopeful about?",
    "pride": "What strengths did you use to achieve this?",
    "relief": "What helped things ease, and is that something you can lean on again?",
    "caring": "How are you caring for yourself as well as for others?",
    "love": "How might you express this appreciation to the people involved?",
}
_GENERIC = ["What do you need most right now - rest, connection, movement, or something else?",
            "What is one small, kind thing you could do for yourself tomorrow?"]


def reflect_on_entry(entry) -> dict:
    check = safety.assess_text(entry.text)
    if check["level"] == "high":
        return _safety_response(check)
    predicted = [e for e in (entry.predicted_emotions or []) if e != "neutral"]
    stress = entry.stress_label == "stress"
    query = entry.text[:600]
    if predicted:
        query += " Feelings: " + ", ".join(predicted[:3])
    if stress:
        query += " Coping with stress."
    chunks = rag.retrieve(query)
    patterns = f"Emotion labels detected in the language: {', '.join(predicted) or 'mostly neutral'}."
    if entry.stress_score is not None:
        patterns += (f" Stress classifier: {'stress-related' if stress else 'no stress-related'} language detected "
                     f"(score {entry.stress_score:.2f}).")
    checkins = ", ".join(f"{k}: {v}" for k, v in (("mood 1-5", entry.mood), ("energy 1-5", entry.energy),
                                                  ("sleep hours", entry.sleep_hours)) if v is not None)
    user = (f"CONTEXT:\n{_context_block(chunks) or '(none)'}\n\nJOURNAL ENTRY:\n\"\"\"{entry.text}\"\"\"\n\n"
            f"MODEL-DERIVED LANGUAGE PATTERNS (non-clinical): {patterns}\n"
            f"SELF-REPORTED CHECK-INS: {checkins or 'none'}\n\n"
            "Write a short, personalised reflection for the author of this entry:\n"
            "1. One or two sentences acknowledging what they wrote, without assuming how they feel.\n"
            "2. Two or three gentle reflective questions specific to the entry.\n"
            "3. One or two practical, low-effort ideas grounded in the CONTEXT, with citations like [1].\n"
            "Keep it under 200 words. Use markdown with short headings.")
    text, info, err = _generate([ChatTurn("system", SYSTEM_PROMPT), ChatTurn("user", user)])
    if text is not None or err is not None:
        res = _finalise(text, chunks, info, err, "", check)
        if res["mode"] == "llm":
            return res
    # Deterministic, clearly-labelled fallback.
    qs = [_PROMPTS[e] for e in predicted if e in _PROMPTS][:2] + _GENERIC
    lines = ["_Generative AI is not configured, so this reflection uses fixed prompts selected from the detected "
             "emotion labels, plus related passages from the knowledge base._", "", "### Questions to reflect on"]
    lines += [f"- {q}" for q in qs[:3]]
    if chunks:
        lines += ["", "### Related guidance"]
        for i, c in enumerate(chunks[:2]):
            lines.append(f"**[{i + 1}] {c['title']} - {c['section']}**")
            lines.append(c["text"])
    if check["level"] == "elevated":
        lines = [check["message"], ""] + lines
    return {"answer": "\n".join(lines), "mode": "retrieval_only", "provider": info, "sources": _sources(chunks),
            "safety": check, "guardrail_removed": [], "error": err}
