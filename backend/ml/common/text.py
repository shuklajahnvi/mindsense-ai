"""Text preprocessing shared by training and inference.

Two levels are used:
* ``clean_text``  - light normalisation suitable for transformers (they need
  punctuation, casing and stop words, so we only fix encoding artefacts,
  URLs, user mentions and whitespace).
* ``normalize_for_tfidf`` - additionally lower-cases; the TF-IDF vectoriser
  handles tokenisation. Stop words are deliberately kept because negations
  ("not", "no", "never") and pronouns carry emotional signal.
"""
from __future__ import annotations

import html
import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_USER_RE = re.compile(r"/?u/[A-Za-z0-9_-]+|@[A-Za-z0-9_]+")
_SUB_RE = re.compile(r"/?r/[A-Za-z0-9_]+")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = _URL_RE.sub(" [URL] ", t)
    t = _USER_RE.sub(" [USER] ", t)
    t = _SUB_RE.sub(" [SUBREDDIT] ", t)
    return _WS_RE.sub(" ", t).strip()


def normalize_for_tfidf(text: str) -> str:
    return clean_text(text).lower()
