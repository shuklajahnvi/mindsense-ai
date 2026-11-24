"""Recurring-theme identification across journal entries.

Approach (a lightweight variant of the BERTopic idea, chosen because the full
BERTopic stack - UMAP + HDBSCAN - is unstable on the very small corpora a
single user produces):
  1. Sentence-Transformer embeddings for each entry (cached in the DB).
  2. K-Means clustering with k chosen by cosine silhouette score
     (k = 2..~sqrt(n/2)+1, preferring solutions without single-entry clusters).
  3. Class-based TF-IDF (c-TF-IDF, as in BERTopic) over the concatenated text
     of each cluster to obtain descriptive keywords.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

from app.services.embeddings import embed

MIN_ENTRIES = 6
_EXTRA_STOP = {"today", "really", "just", "like", "feel", "feeling", "felt", "got", "get", "think", "know", "day",
               "im", "ive", "dont", "didnt", "bit", "lot", "thing", "things", "went", "going", "time", "did", "make",
               "want", "kind", "way", "maybe", "pretty", "actually", "still", "also", "even", "much", "one", "back"}


def ensure_embeddings(db, entries) -> np.ndarray:
    missing = [e for e in entries if not e.embedding]
    if missing:
        vecs = embed([e.text for e in missing])
        for e, v in zip(missing, vecs):
            e.embedding = [round(float(x), 6) for x in v]
        db.commit()
    return np.asarray([e.embedding for e in entries], dtype=np.float32)


def _ctfidf_keywords(docs_per_cluster: list[str], top_n: int = 6) -> list[list[str]]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stop = list(ENGLISH_STOP_WORDS | _EXTRA_STOP)
    cv = CountVectorizer(stop_words=stop, ngram_range=(1, 1), min_df=1, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    tf = cv.fit_transform(docs_per_cluster).toarray().astype(float)
    words = np.array(cv.get_feature_names_out())
    # c-TF-IDF: tf normalised per class * log(1 + avg words per class / word frequency across classes)
    tf_norm = tf / np.maximum(tf.sum(1, keepdims=True), 1)
    avg_words = tf.sum() / len(docs_per_cluster)
    idf = np.log(1 + avg_words / np.maximum(tf.sum(0), 1))
    scores = tf_norm * idf
    return [list(words[np.argsort(-row)[:top_n]]) for row in scores]


def find_topics(db, entries) -> dict:
    n = len(entries)
    if n < MIN_ENTRIES:
        return {"status": "insufficient_data", "min_entries": MIN_ENTRIES, "n_entries": n, "topics": [],
                "method": None}
    X = ensure_embeddings(db, entries)
    # Silhouette tends to keep rising with k on tiny corpora, so cap k near sqrt(n/2) and skip
    # solutions that contain single-entry clusters (a "recurring" theme needs at least two entries).
    k_max = min(8, n - 1, max(2, int(round(np.sqrt(n / 2))) + 1))
    best, fallback = None, None
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        sc = silhouette_score(X, km.labels_, metric="cosine")
        cand = (sc, k, km)
        if fallback is None or sc > fallback[0]:
            fallback = cand
        if np.bincount(km.labels_).min() >= 2 and (best is None or sc > best[0]):
            best = cand
    sil, k, km = best or fallback
    labels = km.labels_
    docs = [" ".join(entries[i].text for i in np.where(labels == c)[0]) for c in range(k)]
    keywords = _ctfidf_keywords(docs)
    topics = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        members = [entries[i] for i in idx]
        # Representative entries: closest to centroid
        centroid = km.cluster_centers_[c]
        order = idx[np.argsort(-(X[idx] @ centroid))]
        stress = [e.stress_score for e in members if e.stress_score is not None]
        topics.append({
            "topic_id": int(c), "name": ", ".join(keywords[c][:3]), "keywords": keywords[c],
            "n_entries": int(len(idx)), "share": round(len(idx) / n, 3),
            "entry_ids": [int(entries[i].id) for i in idx],
            "representative_entry_ids": [int(entries[i].id) for i in order[:3]],
            "avg_stress_score": round(float(np.mean(stress)), 4) if stress else None,
            "first_seen": min(e.created_at for e in members).strftime("%Y-%m-%d"),
            "last_seen": max(e.created_at for e in members).strftime("%Y-%m-%d"),
        })
    topics.sort(key=lambda t: -t["n_entries"])
    return {"status": "ok", "n_entries": n, "k": int(k), "silhouette_cosine": round(float(sil), 4),
            "method": "Sentence-Transformer embeddings + K-Means (k by silhouette) + c-TF-IDF keywords",
            "topics": topics}
