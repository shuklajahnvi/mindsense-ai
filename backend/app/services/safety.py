"""Safety layer.

1. ``assess_text`` - rule-based screening of user text for language that may
   indicate risk of self-harm, suicide, harm to others or abuse. It is a
   conservative keyword/phrase screen (high recall over precision), NOT a
   clinical risk assessment. When triggered, the UI surfaces crisis resources
   and the generative assistant switches to a fixed supportive message.
2. ``sanitize_generated`` - post-filter for LLM output that removes sentences
   making diagnostic claims or medication recommendations.
3. Fixed non-clinical disclaimer and support resources.
"""
from __future__ import annotations

import re

NON_CLINICAL_NOTICE = (
    "MindSense AI is a self-reflection tool, not a medical or diagnostic service. Its outputs describe patterns "
    "in language and are not an assessment of your mental health."
)

SUPPORT_RESOURCES = [
    {"region": "Emergency", "name": "Local emergency services",
     "contact": "911 (US/Canada), 999 or 112 (UK), 112 (EU/India), 000 (Australia)", "url": None},
    {"region": "United States", "name": "988 Suicide & Crisis Lifeline", "contact": "Call or text 988",
     "url": "https://988lifeline.org/"},
    {"region": "Canada", "name": "9-8-8 Suicide Crisis Helpline", "contact": "Call or text 988",
     "url": "https://988.ca/"},
    {"region": "UK & Ireland", "name": "Samaritans", "contact": "116 123 (free, 24/7)",
     "url": "https://www.samaritans.org/"},
    {"region": "India", "name": "Tele-MANAS", "contact": "14416 or 1-800-891-4416 (free, 24/7)",
     "url": "https://telemanas.mohfw.gov.in/"},
    {"region": "Australia", "name": "Lifeline", "contact": "13 11 14", "url": "https://www.lifeline.org.au/"},
    {"region": "International", "name": "Find A Helpline", "contact": "Directory of free helplines by country",
     "url": "https://findahelpline.com/"},
]

# --- risk phrase patterns -------------------------------------------------------
_HIGH = {
    "suicide_or_self_harm": [
        r"\b(kill|killing|hurt|hurting|harm|harming|cut|cutting)\s+my\s?self\b",
        r"\bsuicid(e|al)\b", r"\bend(ing)?\s+(my|it)\s+(life|all)\b", r"\btake\s+my\s+(own\s+)?life\b",
        r"\b(want|wanna|going|plan(ning)?)\s+to\s+die\b", r"\bbetter\s+off\s+dead\b",
        r"\bdon'?t\s+want\s+to\s+(be\s+alive|live|exist|wake\s+up)\b", r"\bno\s+reason\s+to\s+live\b",
        r"\bself[-\s]?harm(ing)?\b", r"\boverdos(e|ing)\b", r"\bwrite\s+(a|my)\s+(suicide\s+)?note\b",
    ],
    "harm_to_others": [
        r"\b(kill|hurt|harm|attack|shoot|stab)\s+(him|her|them|someone|somebody|people|everyone)\b",
    ],
    "abuse_or_danger": [
        r"\b(he|she|they|my\s+\w+)\s+(hits|hit|beats|beat|chokes|choked)\s+me\b",
        r"\b(being|been|was)\s+(abused|assaulted|raped)\b", r"\bnot\s+safe\s+at\s+home\b",
    ],
}
_ELEVATED = {
    "hopelessness": [
        r"\bhopeless\b", r"\bno\s+point\s+(in\s+)?(anything|living|trying|going\s+on)\b",
        r"\bcan'?t\s+(go\s+on|take\s+(it|this)\s+any\s*more|do\s+this\s+any\s*more|cope)\b",
        r"\bwant\s+(it\s+all\s+)?to\s+(stop|end|disappear)\b", r"\bgive\s+up\s+on\s+(life|everything)\b",
        r"\bnobody\s+would\s+(care|miss\s+me)\b", r"\b(i'?m|i\s+am)\s+a\s+burden\b", r"\bwish\s+i\s+(was|were)n'?t\s+here\b",
    ],
}
_NEGATION = re.compile(r"\b(not|never|no longer|don't|do not|wouldn't|would never)\b[^.!?]{0,20}$", re.I)


def _matches(text: str, groups: dict[str, list[str]]) -> dict[str, int]:
    hits: dict[str, int] = {}
    for cat, pats in groups.items():
        for p in pats:
            for m in re.finditer(p, text, flags=re.I):
                # Ignore clear negations immediately before the phrase, e.g. "I would never hurt myself".
                prefix = text[max(0, m.start() - 30):m.start()]
                if cat != "abuse_or_danger" and _NEGATION.search(prefix):
                    continue
                hits[cat] = hits.get(cat, 0) + 1
    return hits


def assess_text(text: str) -> dict:
    high = _matches(text, _HIGH)
    elevated = _matches(text, _ELEVATED)
    level = "high" if high else ("elevated" if elevated else "none")
    msg = None
    if level == "high":
        msg = ("Some of what you wrote suggests you might be going through something very painful or unsafe. "
               "You deserve support right now. If you are in immediate danger, please contact your local emergency "
               "number. You can also reach a free, confidential crisis line - the options below are available "
               "24/7 in many countries.")
    elif level == "elevated":
        msg = ("It sounds like things may feel really heavy at the moment. You don't have to handle this alone - "
               "talking with someone you trust or a support line can help.")
    return {"level": level, "categories": sorted({*high, *elevated}), "message": msg,
            "resources": SUPPORT_RESOURCES if level != "none" else [],
            "method": "rule-based phrase screen (not a clinical risk assessment)"}


# --- output guardrails -----------------------------------------------------------
_CONDITIONS = (r"depress(ion|ed)|anxiety\s+disorder|generali[sz]ed\s+anxiety|ptsd|post-traumatic|bipolar|adhd|ocd|"
               r"borderline|schizo\w*|eating\s+disorder|anorexi\w*|bulimi\w*|panic\s+disorder|mental\s+illness|"
               r"clinical(ly)?\s+\w+")
_DIAGNOSTIC = [
    rf"\byou\s+(may|might|could|probably|likely|clearly)?\s*(have|suffer\s+from|are\s+suffering\s+from|show\s+signs\s+of|"
    rf"have\s+symptoms\s+of|meet\s+the\s+criteria\s+for)\s+(\w+\s+){{0,2}}({_CONDITIONS})",
    rf"\byou\s+(are|seem|sound)\s+(clinically\s+)?(depressed|bipolar|anxious\s+disorder)",
    r"\b(i|this|the\s+(model|analysis|system))\s+(diagnos\w+|can\s+diagnose)",
    r"\bdiagnosis\s+(is|of)\b",
]
_MEDICATION = [
    r"\b(take|try|start|stop|increase|decrease|use)\s+(\w+\s+){0,3}(medication|meds|antidepressant\w*|ssri\w*|"
    r"benzodiazepine\w*|sleeping\s+pills?|melatonin|xanax|prozac|sertraline|zoloft|\w+zepam|\w+oxetine)\b",
    r"\b\d+\s?(mg|milligrams?)\b",
]


def sanitize_generated(text: str) -> tuple[str, list[str]]:
    """Remove sentences with diagnostic or medication content. Returns (clean_text, removed_reasons)."""
    removed: list[str] = []
    out_lines = []
    for line in text.split("\n"):  # line-wise so markdown lists / paragraphs survive
        kept = []
        for s in re.split(r"(?<=[.!?])\s+", line):
            if any(re.search(p, s, re.I) for p in _DIAGNOSTIC):
                removed.append("diagnostic_claim")
            elif any(re.search(p, s, re.I) for p in _MEDICATION):
                removed.append("medication_advice")
            else:
                kept.append(s)
        joined = " ".join(kept)
        if joined.strip() or not line.strip():
            out_lines.append(joined)
    return "\n".join(out_lines).strip(), removed
