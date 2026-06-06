from urllib.parse import urlparse
from src.config import ENABLE_ONLINE_RAG, APPROVED_DOMAINS
from src.schemas import DocumentChunk, RetrievedChunk
from src.rag.reranker import rerank

class OnlineRetriever:
    def __init__(self):
        self.enabled = ENABLE_ONLINE_RAG

    def _approved(self, url: str) -> bool:
        domain = urlparse(url).netloc.replace("www.", "")
        return any(domain.endswith(d) for d in APPROVED_DOMAINS)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.enabled:
            return []
        results = []
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=10):
                    url = r.get("href", "")
                    if not self._approved(url):
                        continue
                    domain = urlparse(url).netloc.replace("www.", "")
                    chunk = DocumentChunk(
                        chunk_id=f"online_{len(results)}",
                        text=r.get("body", "") or r.get("title", ""),
                        source=url,
                        source_type="online",
                        page=None,
                        lecture_number=None,
                        topic=None,
                        metadata={"title": r.get("title", ""), "url": url, "domain": domain},
                    )
                    results.append(RetrievedChunk(chunk=chunk, semantic_score=0.5, authority_score=0.8))
        except Exception:
            return []
        return rerank(query, results, top_k=top_k)