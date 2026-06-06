from collections import Counter
from src.schemas import DocumentChunk

def build_metadata(chunks: list[DocumentChunk]) -> dict:
    return {
        "num_chunks": len(chunks),
        "sources": dict(Counter(c.source for c in chunks)),
        "topics": dict(Counter(c.topic or "unknown" for c in chunks)),
    }