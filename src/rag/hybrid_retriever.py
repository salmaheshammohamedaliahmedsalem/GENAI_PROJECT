from src.config import TOP_K_FINAL
from src.rag.offline_retriever import OfflineRetriever
from src.rag.online_retriever import OnlineRetriever
from src.rag.reranker import rerank

class HybridRetriever:
    def __init__(self):
        self.offline = OfflineRetriever()
        self.online = OnlineRetriever()
        self.last_status = {}

    def retrieve(self, query: str, mode: str = "offline_only"):
        self.last_status = {"mode": mode, "online": {}, "offline_count": 0, "online_count": 0}
        if mode in {"tool_only", "no_retrieval"}:
            return []
        if mode == "offline_only":
            chunks = self.offline.retrieve(query)
            self.last_status["offline_count"] = len(chunks)
            return chunks
        if mode == "online_only":
            chunks = self.online.retrieve(query)
            self.last_status["online_count"] = len(chunks)
            self.last_status["online"] = self.online.diagnostics()
            return chunks
        if mode == "hybrid":
            offline_chunks = self.offline.retrieve(query)
            online_chunks = self.online.retrieve(query)
            self.last_status["offline_count"] = len(offline_chunks)
            self.last_status["online_count"] = len(online_chunks)
            self.last_status["online"] = self.online.diagnostics()
            return rerank(query, offline_chunks + online_chunks, top_k=TOP_K_FINAL)
        return self.offline.retrieve(query)
