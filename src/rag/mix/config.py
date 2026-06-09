"""Shared configuration constants for the Mix RAG system."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CHROMA_PATH = ROOT_DIR / "data" / "vector_db" / "mix_chroma"

# Chunking
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100

# Embedding
EMBED_MODEL: str = "all-MiniLM-L6-v2"

# Retrieval pool sizes
TOP_K_OFFLINE_BM25: int = 12    # BM25 candidates
TOP_K_OFFLINE_VECTOR: int = 10  # semantic candidates
TOP_K_FINAL: int = 6            # final results returned to caller

# HTTP
HTTP_TIMEOUT: int = 25
MAX_TEXT_CHARS: int = 900

# LRU cache for offline queries
OFFLINE_LRU_CACHE_SIZE: int = 20

# Minimum offline (PDF) chunks guaranteed in the final context
MIN_OFFLINE_IN_CONTEXT: int = 3

# Reranker
DEFAULT_CE_MODEL: str = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# LLMs
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LOCAL_MODEL_ID: str = os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

# Domain authority weights for online sources
TRUSTED_DOMAIN_AUTHORITY: dict[str, float] = {
    "arxiv.org": 0.85,
    "semanticscholar.org": 0.80,
    "aclanthology.org": 0.80,
    "huggingface.co": 0.80,
    "openai.com": 0.80,
    "wikipedia.org": 0.70,
    "github.com": 0.70,
    "stackoverflow.com": 0.65,
    "stackexchange.com": 0.60,
    "youtube.com": 0.55,
}
