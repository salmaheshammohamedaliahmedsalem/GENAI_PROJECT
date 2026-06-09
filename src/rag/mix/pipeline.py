"""
pipeline.py — Query orchestration.

Steps:
  1. Expand and clean the query           (src.rag.mix.query)
  2. Dual offline retrieval               (src.rag.mix.offline.DualOfflineRetriever)
  3. Multi-source online retrieval        (src.rag.mix.online.MultiSourceOnlineRetriever)
  4. Merge and 4-signal rerank            (src.rag.mix.reranker)
  5. Offline-first guarantee              (ensure MIN_OFFLINE_IN_CONTEXT PDF chunks)
  6. LLM answer generation                (src.rag.mix.llm)

Public API
----------
    run_query(query, retriever, mode, sources, per_source, top_k,
              rerank_mode, generate_llm, max_new_tokens, pdf_filename) -> dict
"""
from __future__ import annotations

from src.rag.mix.config import MIN_OFFLINE_IN_CONTEXT, TOP_K_FINAL
from src.rag.mix.llm import generate_answer
from src.rag.mix.online import MultiSourceOnlineRetriever
from src.rag.mix.query import enriched_query
from src.rag.mix.reranker import rerank


def _ensure_offline_presence(
    top: list[dict],
    all_offline: list[dict],
    min_offline: int,
) -> list[dict]:
    """Guarantee at least *min_offline* PDF chunks appear in the final context."""
    offline_in_top = [r for r in top if r["chunk"].get("source_type") == "course_pdf"]
    online_in_top  = [r for r in top if r["chunk"].get("source_type") != "course_pdf"]

    ids_in_top = {r["chunk"]["chunk_id"] for r in offline_in_top}
    offline_reserve = [r for r in all_offline if r["chunk"]["chunk_id"] not in ids_in_top]

    missing = max(0, min_offline - len(offline_in_top))
    extra   = offline_reserve[:missing]

    for _ in range(min(len(extra), len(online_in_top))):
        online_in_top.pop()

    offline_in_top.extend(extra)
    return offline_in_top + online_in_top


def _format_results(results: list[dict], pdf_filename: str | None) -> list[dict]:
    formatted = []
    for i, r in enumerate(results, 1):
        c = r["chunk"]
        if c.get("source_type") == "online":
            prov = (c.get("metadata") or {}).get("provider", "online")
            label = f"[{prov}] {c.get('source', '')}"
        else:
            page_info = f", Page {c['page']}" if c.get("page") else ""
            fname = pdf_filename or "PDF"
            label = f"[PDF: {fname}{page_info}]"

        formatted.append({
            "rank": i,
            "label": label,
            "provider": (c.get("metadata") or {}).get("provider") or c.get("source_type") or "unknown",
            "final_score": round(r.get("final_score", 0.0), 4),
            "relevance_score": round(r["relevance_score"], 4) if r.get("relevance_score") is not None else None,
            "text": c.get("text", ""),
            "url": (c.get("metadata") or {}).get("url") if c.get("source_type") == "online" else None,
            "page": c.get("page"),
        })
    return formatted


def run_query(
    query: str,
    retriever,
    mode: str = "hybrid",
    sources: list[str] | None = None,
    per_source: int = 3,
    top_k: int = TOP_K_FINAL,
    rerank_mode: str = "auto",
    generate_llm: bool = True,
    max_new_tokens: int = 256,
    pdf_filename: str | None = None,
) -> dict:
    """
    Run the full Mix RAG pipeline and return a JSON-ready result dict.

    Parameters
    ----------
    query         : Raw user query string.
    retriever     : DualOfflineRetriever instance (or None for online-only).
    mode          : "hybrid" | "offline_only" | "online_only".
    sources       : Online provider subset (None = all 7).
    per_source    : Max chunks per online provider.
    top_k         : Final results count.
    rerank_mode   : "auto" | "cross-encoder" | "lexical".
    generate_llm  : Whether to run LLM generation after retrieval.
    max_new_tokens: Max tokens for LLM output.
    pdf_filename  : Filename shown in result labels.

    Returns
    -------
    dict with keys: query, mode, search_query, pdf_filename,
                    online_diagnostics, reranker, results, llm_answer.
    """
    search_query, query_terms = enriched_query(query)
    diag: dict = {"reranker": None}

    offline_cands: list[dict] = []
    online_cands: list[dict]  = []
    online_diagnostics: dict  = {}

    if mode in ("hybrid", "offline_only"):
        if retriever and retriever.available:
            offline_cands = retriever.candidates(search_query)
        elif mode == "offline_only":
            return {
                "query": query,
                "mode": mode,
                "search_query": search_query,
                "pdf_filename": pdf_filename,
                "online_diagnostics": {},
                "reranker": diag,
                "results": [],
                "llm_answer": None,
                "error": "No PDF indexed. Upload a PDF before using offline mode.",
            }

    if mode in ("hybrid", "online_only"):
        online_retr = MultiSourceOnlineRetriever(sources=sources, per_source=per_source)
        online_cands = online_retr.candidates(search_query)
        online_diagnostics = online_retr.diagnostics()

    all_cands = offline_cands + online_cands
    reranked = rerank(
        query, all_cands,
        top_k=top_k,
        rerank_mode=rerank_mode,
        query_terms=query_terms,
        diag=diag,
    )

    if mode in ("hybrid", "offline_only") and offline_cands:
        reranked = _ensure_offline_presence(reranked, offline_cands, MIN_OFFLINE_IN_CONTEXT)

    formatted = _format_results(reranked, pdf_filename)

    llm_answer = None
    if generate_llm:
        llm_answer = generate_answer(query, reranked, max_new_tokens=max_new_tokens)

    return {
        "query": query,
        "mode": mode,
        "search_query": search_query,
        "pdf_filename": pdf_filename,
        "online_diagnostics": online_diagnostics,
        "reranker": diag,
        "results": formatted,
        "llm_answer": llm_answer,
    }
