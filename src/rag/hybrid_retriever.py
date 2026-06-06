from src.config import TOP_K_FINAL
from src.rag.offline_retriever import OfflineRetriever
from src.rag.online_retriever import OnlineRetriever
from src.rag.reranker import rerank

class HybridRetriever:
    def __init__(self):
        self.offline = OfflineRetriever()
        self.online = OnlineRetriever()

    def retrieve(self, query: str, mode: str = "offline_only"):
        if mode in {"tool_only", "no_retrieval"}:
            return []
        if mode == "offline_only":
            return self.offline.retrieve(query)
        if mode == "online_only":
            return self.online.retrieve(query)
        if mode == "hybrid":
            return rerank(query, self.offline.retrieve(query) + self.online.retrieve(query), top_k=TOP_K_FINAL)
        return self.offline.retrieve(query)