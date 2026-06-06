import re
from src.schemas import RetrievedChunk

def citation_label(r: RetrievedChunk) -> str:
    page = f", page {r.chunk.page}" if r.chunk.page else ""
    return f"[Source: {r.chunk.source}{page}, chunk {r.chunk.chunk_id}]"

def format_sources_for_prompt(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, r in enumerate(chunks, start=1):
        lines.append(f"Source {i}: {citation_label(r)}\n{r.chunk.text}")
    return "\n\n".join(lines)

def list_source_labels(chunks: list[RetrievedChunk]) -> list[str]:
    return [citation_label(r) for r in chunks]

def validate_citations(answer: str, chunks: list[RetrievedChunk]) -> dict:
    valid_labels = set(list_source_labels(chunks))
    cited = set(re.findall(r"\[Source:[^\]]+\]", answer))
    unsupported = [c for c in cited if c not in valid_labels and c != "[Source: retrieved chunk]"]
    return {
        "valid": len(unsupported) == 0,
        "cited": sorted(cited),
        "unsupported": unsupported,
        "available": sorted(valid_labels),
    }