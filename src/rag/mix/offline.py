"""
offline.py — Dual offline retrieval: BM25 keyword search + ChromaDB semantic search.

Both retrievers run on the same PDF chunks. Their results are unioned so the
reranker sees candidates that BM25 found via keyword overlap AND candidates that
semantic search found via meaning — covering what either alone would miss.

Ingest pipeline (once per uploaded PDF):
    parse_pdf → chunk_pages → build_bm25_index + build_vector_index

Retrieval (per query, with LRU cache):
    DualOfflineRetriever.candidates() runs both indexes and merges the union.

Public API
----------
    ingest_pdf(file_path, collection_name) -> dict
    DualOfflineRetriever(collection_name)
        .available                   : bool
        .candidates(query, top_k)    -> list[dict]
"""
from __future__ import annotations

import os
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Optional

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import fitz  # PyMuPDF
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.rag.mix.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_PATH,
    EMBED_MODEL,
    OFFLINE_LRU_CACHE_SIZE,
    TOP_K_OFFLINE_BM25,
    TOP_K_OFFLINE_VECTOR,
)
from src.rag.mix.reranker import _make_record

_embed_model: Optional[SentenceTransformer] = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _bm25_path(collection_name: str) -> Path:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return CHROMA_PATH / f"{collection_name}_bm25.pkl"


# ---------------------------------------------------------------------------
# Ingest helpers
# ---------------------------------------------------------------------------

def _parse_pdf(file_path: str) -> list[str]:
    """Extract non-blank page text from a PDF."""
    doc = fitz.open(str(file_path))
    pages: list[str] = []
    for page in doc:
        text = page.get_text()
        if len(text.strip()) >= 30:
            pages.append(text)
    doc.close()
    return pages


def _chunk_pages(pages: list[str]) -> list[dict]:
    """Split pages into overlapping chunks and return them as normalised dicts."""
    chunks: list[dict] = []
    chunk_idx = 0
    for page_idx, text in enumerate(pages):
        text = " ".join(text.split())
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            snippet = text[start:end]
            if end < len(text):
                last_space = snippet.rfind(" ")
                if last_space > 0 and len(snippet) - last_space < 80:
                    end = start + last_space
                    snippet = text[start:end]
            chunks.append({
                "chunk_id": f"pdf_page_{page_idx + 1}_{chunk_idx}",
                "text": snippet,
                "source": "Uploaded PDF",
                "source_type": "course_pdf",
                "page": page_idx + 1,
                "lecture_number": None,
                "topic": None,
                "metadata": {
                    "title": f"Page {page_idx + 1}",
                    "url": None,
                    "domain": "local_pdf",
                    "provider": "PDF File",
                },
            })
            chunk_idx += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _build_bm25_index(chunks: list[dict], collection_name: str) -> None:
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with _bm25_path(collection_name).open("wb") as fh:
        pickle.dump({"bm25": bm25, "chunks": chunks}, fh)


def _build_vector_index(chunks: list[dict], collection_name: str) -> None:
    model = _get_embed_model()
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=texts,
        embeddings=np.array(vectors, dtype=np.float32).tolist(),
        ids=[c["chunk_id"] for c in chunks],
        metadatas=[{"page": c["page"]} for c in chunks],
    )


# ---------------------------------------------------------------------------
# Public ingest entry point
# ---------------------------------------------------------------------------

def ingest_pdf(file_path: str, collection_name: str = "mix_docs") -> dict:
    """Parse, chunk, and index a PDF into both BM25 and ChromaDB."""
    pages = _parse_pdf(file_path)
    if not pages:
        return {"status": "error", "num_chunks": 0, "collection": collection_name}
    chunks = _chunk_pages(pages)
    _build_bm25_index(chunks, collection_name)
    _build_vector_index(chunks, collection_name)
    return {"status": "ok", "num_chunks": len(chunks), "collection": collection_name}


# ---------------------------------------------------------------------------
# DualOfflineRetriever
# ---------------------------------------------------------------------------

class DualOfflineRetriever:
    """
    Runs BM25 and ChromaDB semantic search in parallel on the same PDF chunks.

    BM25 catches exact keyword matches; semantic search catches paraphrases
    and synonyms. The union of both result sets is returned as candidates
    for the reranker, which then scores them with its 4-signal blend.

    Results for identical queries are cached (LRU, capacity OFFLINE_LRU_CACHE_SIZE).
    """

    def __init__(self, collection_name: str) -> None:
        self._collection_name = collection_name
        self._bm25_data: dict | None = None
        self._chroma_collection: chromadb.Collection | None = None
        self._cache: OrderedDict[str, list[dict]] = OrderedDict()

        p = _bm25_path(collection_name)
        if p.exists():
            with p.open("rb") as fh:
                self._bm25_data = pickle.load(fh)

        try:
            client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            self._chroma_collection = client.get_collection(collection_name)
        except Exception:
            self._chroma_collection = None

    @property
    def available(self) -> bool:
        return self._bm25_data is not None or self._chroma_collection is not None

    def candidates(self, search_query: str, top_k: int = TOP_K_OFFLINE_BM25) -> list[dict]:
        """Return the union of BM25 and semantic candidates (LRU-cached)."""
        if search_query in self._cache:
            return self._cache[search_query]

        bm25_records = self._bm25_candidates(search_query)
        vector_records = self._vector_candidates(search_query)

        merged: dict[str, dict] = {}
        for rec in bm25_records:
            merged[rec["chunk"]["chunk_id"]] = rec
        for rec in vector_records:
            cid = rec["chunk"]["chunk_id"]
            if cid not in merged:
                merged[cid] = rec

        result = list(merged.values())

        if len(self._cache) >= OFFLINE_LRU_CACHE_SIZE:
            self._cache.popitem(last=False)
        self._cache[search_query] = result
        return result

    def _bm25_candidates(self, search_query: str) -> list[dict]:
        if self._bm25_data is None:
            return []
        bm25 = self._bm25_data["bm25"]
        chunks = self._bm25_data["chunks"]
        tokens = search_query.split() or ["query"]
        scores = bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K_OFFLINE_BM25]
        seen: dict[str, dict] = {}
        for i in top_idx:
            chunk = chunks[i]
            cid = chunk["chunk_id"]
            rec = seen.setdefault(cid, _make_record(chunk))
            rec["keyword_score"] = float(scores[i])
        return list(seen.values())

    def _vector_candidates(self, search_query: str) -> list[dict]:
        if self._chroma_collection is None:
            return []
        try:
            count = self._chroma_collection.count()
            if count == 0:
                return []
            model = _get_embed_model()
            query_vec = model.encode(
                [search_query], show_progress_bar=False, convert_to_numpy=True
            )[0]
            k = min(TOP_K_OFFLINE_VECTOR, count)
            results = self._chroma_collection.query(
                query_embeddings=[query_vec.tolist()],
                n_results=k,
                include=["documents", "distances", "metadatas"],
            )
            records: list[dict] = []
            metadatas = results["metadatas"][0] or [{}] * k
            for chunk_id, doc, dist, meta in zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0],
                metadatas,
            ):
                page = (meta or {}).get("page")
                chunk = {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "source": "Uploaded PDF",
                    "source_type": "course_pdf",
                    "page": page,
                    "lecture_number": None,
                    "topic": None,
                    "metadata": {
                        "title": f"Page {page}" if page else "PDF",
                        "url": None,
                        "domain": "local_pdf",
                        "provider": "PDF File",
                    },
                }
                rec = _make_record(chunk)
                rec["keyword_score"] = 0.0
                records.append(rec)
            return records
        except Exception:
            return []
