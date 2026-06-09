"""Runtime PDF ingestion: parse, chunk, embed, and index a student-uploaded PDF.

Ported from the companion hybrid-RAG demo. Requires:
  pymupdf, sentence-transformers, langchain-text-splitters, chromadb
Install via: pip install -r requirements_semantic.txt pymupdf langchain-text-splitters
"""
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from src.config import VECTOR_DB_DIR
from src.schemas import DocumentChunk, RetrievedChunk

_CHUNK_SIZE = 300
_CHUNK_OVERLAP = 50
_EMBED_MODEL = "all-MiniLM-L6-v2"
_CACHE_SIZE = 10

_sentence_model = None
_retrieve_cache: OrderedDict[str, list[RetrievedChunk]] = OrderedDict()


def _get_embed_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer(_EMBED_MODEL)
    return _sentence_model


def parse_pdf(file_path: str | Path) -> list[str]:
    """Extract text from each page via PyMuPDF. Skips near-empty pages."""
    import fitz  # PyMuPDF
    doc = fitz.open(str(file_path))
    pages = []
    for page in doc:
        text = page.get_text()
        if len(text.strip()) >= 30:
            pages.append(text)
    doc.close()
    return pages


def chunk_text(pages: list[str]) -> list[str]:
    """Join all pages then split into overlapping fixed-size chunks."""
    if not pages:
        return []
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)
    return splitter.split_text("\n\n".join(pages))


def embed_chunks(chunks: list[str]):
    """Embed chunks with SentenceTransformer; returns float32 ndarray."""
    import numpy as np
    embeddings = _get_embed_model().encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    return np.array(embeddings, dtype=np.float32)


def build_session_index(
    chunks: list[str],
    vectors,
    collection_name: str,
    source_name: str,
):
    """Create (or replace) a ChromaDB collection with pre-computed embeddings."""
    import chromadb
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=chunks,
        embeddings=vectors.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[
            {"source": source_name, "source_type": "uploaded_pdf", "part_index": i}
            for i in range(len(chunks))
        ],
    )
    return collection


def ingest_pdf(file_path: str | Path, collection_name: str = "session_pdf") -> dict:
    """Full pipeline: parse → chunk → embed → index. Returns a status dict."""
    pages = parse_pdf(file_path)
    chunks = chunk_text(pages)
    if not chunks:
        return {"status": "error", "message": "No readable text found in PDF.", "num_chunks": 0}
    vectors = embed_chunks(chunks)
    build_session_index(chunks, vectors, collection_name, source_name=Path(file_path).name)
    _retrieve_cache.clear()
    return {
        "status": "ok",
        "num_chunks": len(chunks),
        "collection": collection_name,
        "filename": Path(file_path).name,
    }


def retrieve_from_session(
    question: str,
    collection_name: str = "session_pdf",
    k: int = 5,
) -> list[RetrievedChunk]:
    """Query the session-indexed PDF with an in-memory LRU cache."""
    cache_key = f"{collection_name}::{question}"
    if cache_key in _retrieve_cache:
        _retrieve_cache.move_to_end(cache_key)
        return _retrieve_cache[cache_key]

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        collection = client.get_collection(collection_name)
    except Exception:
        return []

    count = collection.count()
    if count == 0:
        return []

    query_vec = _get_embed_model().encode(
        [question], show_progress_bar=False, convert_to_numpy=True
    )[0]
    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(k, count),
        include=["documents", "distances", "metadatas"],
    )

    output = []
    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results.get("metadatas", [[]])[0]
    for i, (doc, dist, meta) in enumerate(zip(docs, dists, metas)):
        chunk = DocumentChunk(
            chunk_id=f"{collection_name}_chunk_{i}",
            text=doc,
            source=meta.get("source", collection_name),
            source_type="uploaded_pdf",
            page=None,
            lecture_number=None,
            topic=None,
            metadata=meta,
        )
        output.append(RetrievedChunk(chunk=chunk, semantic_score=max(0.0, 1.0 - float(dist))))

    if len(_retrieve_cache) >= _CACHE_SIZE:
        _retrieve_cache.popitem(last=False)
    _retrieve_cache[cache_key] = output
    return output


def is_available() -> bool:
    """True when all required packages for PDF ingestion are importable."""
    try:
        import fitz  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401
        import chromadb  # noqa: F401
        return True
    except ImportError:
        return False
