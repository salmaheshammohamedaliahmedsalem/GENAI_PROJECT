"""
reranker.py — 4-signal relevance scoring and re-ranking.

Supports two relevance signals:
  - Cross-encoder neural reranker (sentence-transformers, optional, lazy-loaded).
  - Lexical relevance fallback (always available, no extra dependencies).

Blends relevance with keyword BM25 score, metadata bonus, and domain authority.

Final score weights:
  With cross-encoder : 0.65 * relevance + 0.10 * keyword + 0.10 * metadata + 0.15 * authority
  Lexical fallback   : 0.50 * relevance + 0.25 * keyword + 0.10 * metadata + 0.15 * authority

Public API
----------
    rerank(query, retrieved, top_k, rerank_mode, query_terms, ce_model, diag) -> list[dict]
    _make_record(chunk, semantic_score, keyword_score)                         -> dict
    _online_chunk(provider, idx, title, body, url)                             -> dict
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from src.rag.mix.config import (
    DEFAULT_CE_MODEL,
    MAX_TEXT_CHARS,
    TOP_K_FINAL,
    TRUSTED_DOMAIN_AUTHORITY,
)
from src.rag.mix.query import clean_query

_CE_MODEL = None
_CE_LOAD_ERROR = ""


def _get_cross_encoder(model_name: str):
    global _CE_MODEL, _CE_LOAD_ERROR
    if _CE_MODEL is not None:
        return _CE_MODEL
    if _CE_LOAD_ERROR:
        return None
    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        _CE_MODEL = CrossEncoder(model_name)
        return _CE_MODEL
    except Exception as e:
        _CE_LOAD_ERROR = f"{type(e).__name__}: {e}"
        return None


# ---------------------------------------------------------------------------
# Relevance helpers
# ---------------------------------------------------------------------------

def lexical_relevance(query_terms: list[str], text: str, title: str = "") -> float:
    """Token-overlap relevance in [0, 1] with a phrase-match bonus (+0.2)."""
    if not query_terms:
        return 0.0
    blob = f"{title} {text}".lower()
    blob_tokens = set(re.findall(r"[a-z0-9][a-z0-9\-]*", blob))
    uniq = set(query_terms)
    hits = sum(1 for t in uniq if t in blob_tokens)
    coverage = hits / len(uniq)
    phrase = " ".join(query_terms)
    bonus = 0.2 if phrase and phrase in blob else 0.0
    return min(1.0, coverage + bonus)


def _norm(value, max_value: float = 1.0) -> float:
    if value is None or max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def metadata_score(query: str, chunk: dict) -> float:
    """Metadata bonus in [0, 1] for lecture/topic/RAG-specific signals."""
    q = query.lower()
    text = (chunk.get("text") or "").lower()
    topic = (chunk.get("topic") or "").lower()
    score = 0.0
    if "project" in q and chunk.get("source_type") == "project_doc":
        score += 0.5
    if chunk.get("lecture_number") and f"lecture {chunk['lecture_number']}" in q:
        score += 0.4
    if topic and any(w in q for w in topic.split()):
        score += 0.3
    if "rag" in q and "retrieval-augmented generation" in topic:
        score += 0.4
        if "retrieval augmented generation" in text or "retrieval-augmented generation" in text:
            score += 0.35
        if any(p in text for p in ["architecture overview", "relevant context", "grounds",
                                   "advantages of rag", "retriever", "knowledge base"]):
            score += 0.4
    return min(score, 1.0)


def authority_score(chunk: dict) -> float:
    """Domain authority in [0, 1]; course PDFs rank highest."""
    st = chunk.get("source_type")
    if st in {"course_pdf", "project_doc"}:
        return 1.0
    if st == "online":
        domain = (chunk.get("metadata") or {}).get("domain", "")
        for d, a in TRUSTED_DOMAIN_AUTHORITY.items():
            if domain == d or domain.endswith("." + d):
                return a
        return 0.4
    return 0.5


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------

def _make_record(chunk: dict, semantic_score=None, keyword_score=None) -> dict:
    return {
        "chunk": chunk,
        "semantic_score": semantic_score,
        "keyword_score": keyword_score,
    }


def _online_chunk(provider: str, idx: int, title: str, body: str, url: str) -> dict:
    text = " ".join((body or title or "").split())[:MAX_TEXT_CHARS]
    domain = urlparse(url).netloc.replace("www.", "").lower() if url else ""
    return {
        "chunk_id": f"online_{provider}_{idx}",
        "text": text,
        "source": url or provider,
        "source_type": "online",
        "page": None,
        "lecture_number": None,
        "topic": None,
        "metadata": {
            "title": title,
            "url": url,
            "domain": domain,
            "provider": provider,
        },
    }


# ---------------------------------------------------------------------------
# Main rerank function
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    retrieved: list[dict],
    top_k: int = TOP_K_FINAL,
    rerank_mode: str = "auto",
    query_terms: list[str] | None = None,
    ce_model: str = DEFAULT_CE_MODEL,
    diag: dict | None = None,
) -> list[dict]:
    """
    Score and sort *retrieved* records by a 4-signal blend.

    Parameters
    ----------
    query        : Original user query (for metadata scoring).
    retrieved    : Candidate records from DualOfflineRetriever / MultiSourceOnlineRetriever.
    top_k        : Maximum results to return.
    rerank_mode  : "auto" | "cross-encoder" | "lexical".
    query_terms  : Pre-tokenised terms (computed from query if None).
    ce_model     : Cross-encoder model name.
    diag         : Optional dict to write diagnostics into.

    Returns
    -------
    list[dict] — top_k records with final_score, relevance_score,
                 metadata_score, authority_score set.
    """
    if not retrieved:
        return []
    if query_terms is None:
        query_terms = clean_query(query).split()

    used_ce = False
    if rerank_mode in ("auto", "cross-encoder"):
        ce = _get_cross_encoder(ce_model)
        if ce is not None:
            try:
                pairs = [
                    (query, (r["chunk"].get("text") or
                             (r["chunk"].get("metadata") or {}).get("title", "")))
                    for r in retrieved
                ]
                raw = [float(s) for s in ce.predict(pairs)]
                for r, s in zip(retrieved, _minmax(raw)):
                    r["relevance_score"] = s
                used_ce = True
            except Exception as e:
                if diag is not None:
                    diag["reranker_error"] = f"{type(e).__name__}: {e}"
        elif rerank_mode == "cross-encoder" and diag is not None:
            diag["reranker_error"] = _CE_LOAD_ERROR or "sentence-transformers not installed"

    if not used_ce:
        for r in retrieved:
            c = r["chunk"]
            r["relevance_score"] = lexical_relevance(
                query_terms, c.get("text", ""),
                (c.get("metadata") or {}).get("title", ""),
            )

    if diag is not None:
        diag["reranker"] = "cross-encoder" if used_ce else "lexical"
        if used_ce:
            diag["reranker_model"] = ce_model

    max_key = max([r.get("keyword_score") or 0 for r in retrieved] + [1])
    for r in retrieved:
        chunk = r["chunk"]
        rel = r.get("relevance_score", 0.0)
        key = _norm(r.get("keyword_score"), max_key)
        meta = metadata_score(query, chunk)
        auth = authority_score(chunk)
        r["metadata_score"] = meta
        r["authority_score"] = auth
        if used_ce:
            r["final_score"] = 0.65 * rel + 0.10 * key + 0.10 * meta + 0.15 * auth
        else:
            r["final_score"] = 0.50 * rel + 0.25 * key + 0.10 * meta + 0.15 * auth

    return sorted(retrieved, key=lambda x: x["final_score"], reverse=True)[:top_k]
