"""
query.py — Query cleaning and NLP-domain synonym expansion.

Pipeline:
  1. Strip multi-word filler phrases ("explain", "tell me about", ...).
  2. Remove stopwords and deduplicate tokens.
  3. Expand domain abbreviations (rag → retrieval augmented generation, ...).

Public API
----------
    clean_query(q)                             -> str
    expand_query(cleaned)                      -> list[str]
    enriched_query(q, clean, expand)           -> tuple[str, list[str]]
"""
from __future__ import annotations

import re

STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of",
    "to", "in", "on", "for", "and", "or", "with", "about", "into", "from", "by",
    "this", "that", "these", "those", "it", "its", "as", "at", "do", "does",
    "did", "can", "could", "should", "would", "i", "me", "my", "we", "our",
    "you", "your", "please", "some", "any", "more", "based", "using", "use",
}

# Longer phrases before shorter prefixes to avoid partial matches.
FILLER_PHRASES: list[str] = [
    "based on our course lectures", "from our lecture", "from the lectures",
    "in our course", "based on the course", "tell me about", "tell me",
    "what is", "what are", "what's", "how does", "how do", "how can i",
    "give me", "show me", "teach me", "explain to me", "explain", "describe",
    "can you", "i want to", "i need to", "help me", "according to",
]

EXPANSIONS: dict[str, list[str]] = {
    "rag":          ["retrieval augmented generation"],
    "hybrid":       ["hybrid retrieval", "dense sparse retrieval"],
    "search":       ["retrieval"],
    "lora":         ["low-rank adaptation", "parameter efficient fine-tuning"],
    "peft":         ["parameter efficient fine-tuning"],
    "embedding":    ["dense vector retrieval"],
    "embeddings":   ["dense vector retrieval"],
    "vector":       ["vector database similarity search"],
    "rerank":       ["cross encoder reranking"],
    "reranking":    ["cross encoder reranking"],
    "transformer":  ["self-attention architecture"],
    "attention":    ["self-attention transformer"],
    "rlhf":         ["reinforcement learning from human feedback"],
    "dpo":          ["direct preference optimization"],
    "agent":        ["llm agents tool use"],
    "agents":       ["llm agents tool use"],
    "quantization": ["model quantization int8 int4"],
}


def clean_query(q: str) -> str:
    """Strip filler phrases and stopwords; return a clean keyword string."""
    s = " " + q.lower() + " "
    for phrase in FILLER_PHRASES:
        s = s.replace(" " + phrase + " ", " ")
    toks = re.findall(r"[a-z0-9][a-z0-9\-]*", s)
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    cleaned = " ".join(dict.fromkeys(toks))
    return cleaned or q.lower().strip()


def expand_query(cleaned: str) -> list[str]:
    """Return domain synonym expansion strings for the cleaned query."""
    extra: list[str] = []
    for t in cleaned.split():
        for exp in EXPANSIONS.get(t, []):
            if exp not in extra:
                extra.append(exp)
    return extra


def enriched_query(
    query: str, clean: bool = True, expand: bool = True
) -> tuple[str, list[str]]:
    """
    Build the final search string and key terms for lexical relevance scoring.

    Returns
    -------
    (search_string, query_terms)
        search_string – enriched query sent to every retrieval provider
        query_terms   – individual tokens used for lexical relevance scoring
    """
    base = clean_query(query) if clean else query.lower().strip()
    expansions = expand_query(base) if expand else []
    search = base if not expansions else f"{base} {' '.join(expansions)}"
    return search.strip(), base.split()
