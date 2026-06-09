from src.config import TOP_K_FINAL
from src.rag.offline_retriever import OfflineRetriever
from src.rag.reranker import rerank
from src.schemas import DocumentChunk, RetrievedChunk


def _mix_record_to_retrieved_chunk(record: dict) -> RetrievedChunk:
    """Convert a mix RAG record dict to a project RetrievedChunk."""
    c = record["chunk"]
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=c["chunk_id"],
            text=c.get("text", ""),
            source=c.get("source", ""),
            source_type=c.get("source_type", "online"),
            page=c.get("page"),
            lecture_number=c.get("lecture_number"),
            topic=c.get("topic"),
            metadata=c.get("metadata") or {},
        ),
        semantic_score=record.get("relevance_score"),
        keyword_score=record.get("keyword_score"),
        metadata_score=record.get("metadata_score"),
        authority_score=record.get("authority_score"),
        final_score=record.get("final_score", 0.0),
    )


def _enriched_query(query: str) -> str:
    """Apply mix query expansion; fall back to original query on import error."""
    try:
        from src.rag.mix.query import enriched_query
        search_query, _ = enriched_query(query)
        return search_query
    except Exception:
        return query


def _online_candidates(search_query: str) -> tuple[list[RetrievedChunk], dict]:
    """Retrieve from mix's 7-source online retriever; return chunks + diagnostics."""
    try:
        from src.rag.mix.online import MultiSourceOnlineRetriever
        retr = MultiSourceOnlineRetriever()
        records = retr.candidates(search_query)
        chunks = [_mix_record_to_retrieved_chunk(r) for r in records]
        return chunks, retr.diagnostics()
    except Exception as exc:
        return [], {"error": str(exc), "providers": []}


class HybridRetriever:
    def __init__(self):
        self.offline = OfflineRetriever()
        self.last_status = {}

    def retrieve(self, query: str, mode: str = "offline_only") -> list[RetrievedChunk]:
        self.last_status = {"mode": mode, "online": {}, "offline_count": 0, "online_count": 0}
        if mode in {"tool_only", "no_retrieval"}:
            return []

        search_query = _enriched_query(query)

        if mode == "offline_only":
            chunks = self.offline.retrieve(search_query)
            self.last_status["offline_count"] = len(chunks)
            return chunks

        if mode == "online_only":
            online_chunks, diag = _online_candidates(search_query)
            self.last_status["online"] = diag
            ranked = rerank(query, online_chunks, top_k=TOP_K_FINAL)
            self.last_status["online_count"] = len(ranked)
            return ranked

        if mode == "hybrid":
            offline_chunks = self.offline.retrieve(search_query)
            online_chunks, diag = _online_candidates(search_query)
            self.last_status["offline_count"] = len(offline_chunks)
            self.last_status["online_count"] = len(online_chunks)
            self.last_status["online"] = diag
            return rerank(query, offline_chunks + online_chunks, top_k=TOP_K_FINAL)

        return self.offline.retrieve(search_query)
