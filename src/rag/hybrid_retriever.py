from src.config import TOP_K_FINAL
from src.rag.offline_retriever import OfflineRetriever
from src.rag.online_retriever import OnlineRetriever
from src.rag.reranker import rerank


class HybridRetriever:
    def __init__(self):
        self.offline = OfflineRetriever()
        self.online = OnlineRetriever()
        self.last_status = {}

    def _get_session_chunks(self, query: str, session_collection: str | None) -> list:
        if not session_collection:
            return []
        try:
            from src.rag.pdf_ingestor import retrieve_from_session
            return retrieve_from_session(query, collection_name=session_collection, k=5)
        except Exception:
            return []

    def retrieve(
        self,
        query: str,
        mode: str = "offline_only",
        session_collection: str | None = None,
    ) -> list:
        self.last_status = {
            "mode": mode,
            "online": {},
            "offline_count": 0,
            "online_count": 0,
            "session_count": 0,
        }
        if mode in {"tool_only", "no_retrieval"}:
            return []

        session_chunks = self._get_session_chunks(query, session_collection)
        self.last_status["session_count"] = len(session_chunks)

        if mode == "offline_only":
            course_chunks = self.offline.retrieve(query)
            self.last_status["offline_count"] = len(course_chunks)
            all_chunks = session_chunks + course_chunks
            return rerank(query, all_chunks, top_k=TOP_K_FINAL) if session_chunks else course_chunks

        if mode == "online_only":
            online_chunks = self.online.retrieve(query)
            self.last_status["online_count"] = len(online_chunks)
            self.last_status["online"] = self.online.diagnostics()
            all_chunks = session_chunks + online_chunks
            return rerank(query, all_chunks, top_k=TOP_K_FINAL) if session_chunks else online_chunks

        if mode == "hybrid":
            offline_chunks = self.offline.retrieve(query)
            online_chunks = self.online.retrieve(query)
            self.last_status["offline_count"] = len(offline_chunks)
            self.last_status["online_count"] = len(online_chunks)
            self.last_status["online"] = self.online.diagnostics()
            return rerank(query, session_chunks + offline_chunks + online_chunks, top_k=TOP_K_FINAL)

        course_chunks = self.offline.retrieve(query)
        all_chunks = session_chunks + course_chunks
        return rerank(query, all_chunks, top_k=TOP_K_FINAL) if session_chunks else course_chunks
