from src.schemas import RetrievedChunk

def _norm(value: float | None, max_value: float = 1.0) -> float:
    if value is None:
        return 0.0
    if max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))

def metadata_score(query: str, chunk) -> float:
    q = query.lower()
    score = 0.0
    if "project" in q and chunk.source_type == "project_doc":
        score += 0.5
    if chunk.lecture_number and f"lecture {chunk.lecture_number}" in q:
        score += 0.4
    if chunk.topic and any(word in q for word in chunk.topic.lower().split()):
        score += 0.3
    return min(score, 1.0)

def authority_score(chunk) -> float:
    if chunk.source_type in {"course_pdf", "project_doc"}:
        return 1.0
    if chunk.source_type == "online":
        domain = chunk.metadata.get("domain", "")
        if any(x in domain for x in ["openai.com", "huggingface.co", "arxiv.org", "aclanthology.org"]):
            return 0.8
        return 0.4
    return 0.5

def rerank(query: str, retrieved: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
    max_sem = max([r.semantic_score or 0 for r in retrieved] + [1])
    max_key = max([r.keyword_score or 0 for r in retrieved] + [1])
    for r in retrieved:
        r.metadata_score = metadata_score(query, r.chunk)
        r.authority_score = authority_score(r.chunk)
        r.final_score = (
            0.45 * _norm(r.semantic_score, max_sem)
            + 0.25 * _norm(r.keyword_score, max_key)
            + 0.15 * (r.metadata_score or 0)
            + 0.15 * (r.authority_score or 0)
        )
    return sorted(retrieved, key=lambda x: x.final_score, reverse=True)[:top_k]