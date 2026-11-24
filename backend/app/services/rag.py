"""Retrieval-Augmented Generation over the curated wellbeing knowledge base.

Pipeline: load markdown documents (with source metadata in front matter) ->
split into heading-aware, overlapping chunks -> embed with Sentence
Transformers -> store in a persistent ChromaDB collection (cosine space) ->
retrieve top-k chunks for a query -> pass them as numbered context to the LLM.
The index is rebuilt automatically when the knowledge-base files change.
"""
from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path

import yaml

from app.config import get_settings
from app.services.embeddings import embed

COLLECTION = "wellbeing_kb"
CHUNKER_VERSION = "2"  # bump when chunking changes so the index is rebuilt
_lock = threading.Lock()
_collection = None


def _parse_doc(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, flags=re.S)
    if not m:
        return {"title": path.stem, "topic": path.stem, "sources": []}, raw
    return yaml.safe_load(m.group(1)), m.group(2)


def _blocks(content: str, max_words: int) -> list[str]:
    """Paragraphs, with markdown lists split into individual items; over-long blocks hard-split."""
    out: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", content)):
        if not para:
            continue
        items = re.split(r"\n(?=\s*(?:[-*]|\d+\.)\s)", para) if re.match(r"\s*(?:[-*]|\d+\.)\s", para) else [para]
        for item in items:
            words = item.split()
            while len(words) > max_words:
                out.append(" ".join(words[:max_words]))
                words = words[max_words:]
            if words:
                out.append(item.strip() if len(item.split()) == len(words) else " ".join(words))
    return out


def chunk_document(body: str, max_words: int = 170, overlap_words: int = 30) -> list[tuple[str, str]]:
    """Split by '## ' sections, then pack paragraphs / list items into ~max_words chunks.

    Formatting (line breaks, list items) is preserved. Consecutive chunks overlap by the last block of
    the previous chunk when that block is short (<= 2 * overlap_words words).
    Returns [(section_heading, chunk_text)].
    """
    chunks: list[tuple[str, str]] = []
    for sec in re.split(r"^##\s+", body, flags=re.M):
        sec = sec.strip()
        if not sec:
            continue
        heading, _, content = sec.partition("\n")
        buf: list[str] = []
        n_words = 0
        for block in _blocks(content, max_words):
            w = len(block.split())
            if buf and n_words + w > max_words:
                chunks.append((heading.strip(), "\n".join(buf)))
                last = buf[-1]
                buf = [last] if len(last.split()) <= 2 * overlap_words else []
                n_words = sum(len(b.split()) for b in buf)
            buf.append(block)
            n_words += w
        if buf:
            chunks.append((heading.strip(), "\n".join(buf)))
    return chunks


def _kb_files() -> list[Path]:
    return sorted(Path(get_settings().knowledge_base_dir).glob("*.md"))


def _kb_hash() -> str:
    h = hashlib.sha256()
    for f in _kb_files():
        h.update(f.name.encode()); h.update(f.read_bytes())
    h.update(get_settings().embedding_model.encode())
    h.update(CHUNKER_VERSION.encode())
    return h.hexdigest()[:16]


def build_index(force: bool = False):
    """(Re)build the Chroma collection if the knowledge base changed."""
    import chromadb

    global _collection
    with _lock:
        s = get_settings()
        client = chromadb.PersistentClient(path=s.chroma_dir)
        kb_hash = _kb_hash()
        try:
            col = client.get_collection(COLLECTION)
            if not force and (col.metadata or {}).get("kb_hash") == kb_hash and col.count() > 0:
                _collection = col
                return col
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine", "kb_hash": kb_hash,
                                                             "embedding_model": s.embedding_model})
        ids, docs, metas = [], [], []
        for f in _kb_files():
            meta, body = _parse_doc(f)
            sources = meta.get("sources") or []
            for i, (heading, text) in enumerate(chunk_document(body)):
                ids.append(f"{f.stem}-{i}")
                docs.append(text)
                metas.append({
                    "doc": f.stem, "title": meta.get("title", f.stem), "topic": meta.get("topic", ""),
                    "section": heading,
                    "source_names": " | ".join(x["name"] for x in sources),
                    "source_urls": " | ".join(x["url"] for x in sources),
                })
        # Embed "title - section: text" so headings contribute to retrieval.
        vecs = embed([f"{m['title']} - {m['section']}: {d}" for m, d in zip(metas, docs)])
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=vecs.tolist())
        _collection = col
        return col


def get_collection():
    return _collection if _collection is not None else build_index()


def retrieve(query: str, k: int | None = None) -> list[dict]:
    s = get_settings()
    col = get_collection()
    res = col.query(query_embeddings=embed([query]).tolist(), n_results=k or s.rag_top_k)
    out = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1.0 - float(dist)
        if sim < s.rag_min_similarity:
            continue
        names, urls = meta["source_names"].split(" | "), meta["source_urls"].split(" | ")
        out.append({"id": cid, "title": meta["title"], "section": meta["section"], "topic": meta["topic"],
                    "text": doc, "similarity": round(sim, 4),
                    "sources": [{"name": n, "url": u} for n, u in zip(names, urls) if n]})
    return out


def index_status() -> dict:
    try:
        col = get_collection()
        return {"ready": True, "chunks": col.count(), "documents": len(_kb_files()),
                "embedding_model": (col.metadata or {}).get("embedding_model"), "store": "ChromaDB (persistent)"}
    except Exception as e:  # pragma: no cover - surfaced in /api/status
        return {"ready": False, "error": str(e)}
