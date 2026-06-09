from __future__ import annotations

import re
from urllib.parse import urlparse

from src.config import APPROVED_DOMAINS, ENABLE_ONLINE_RAG, TAVILY_API_KEY
from src.rag.reranker import rerank
from src.schemas import DocumentChunk, RetrievedChunk

# Strip document-self-reference phrases before sending a query to the web.
# Phrases like "from this lecture" are meaningless to a search engine.
_DOC_REF_PATTERNS = re.compile(
    r"\b(of this document|in this document|from this document|"
    r"based on this material|in this (lecture|material|text|slides?)|"
    r"from this (lecture|material|text|slides?))\b",
    re.IGNORECASE,
)


def _web_query(question: str) -> str:
    return _DOC_REF_PATTERNS.sub("", question).strip()


class OnlineRetriever:
    def __init__(self):
        self.enabled = ENABLE_ONLINE_RAG
        self.last_status: dict = self._empty_status()

    def _empty_status(self) -> dict:
        return {
            "enabled": self.enabled,
            "providers": [],
            "raw_results": 0,
            "accepted_results": 0,
            "filtered_results": 0,
            "errors": [],
            "message": "",
        }

    def diagnostics(self) -> dict:
        return self.last_status

    def _approved(self, url: str) -> bool:
        domain = urlparse(url).netloc.replace("www.", "").lower()
        return any(domain == approved or domain.endswith(f".{approved}") for approved in APPROVED_DOMAINS)

    def _domain(self, url: str) -> str:
        return urlparse(url).netloc.replace("www.", "").lower()

    def _append_result(self, results: list[RetrievedChunk], title: str, body: str, url: str, provider: str) -> None:
        if not url or not self._approved(url):
            self.last_status["filtered_results"] += 1
            return

        text = (body or title or "").strip()
        if not text:
            self.last_status["filtered_results"] += 1
            return

        domain = self._domain(url)
        chunk = DocumentChunk(
            chunk_id=f"online_{provider}_{len(results)}",
            text=text,
            source=url,
            source_type="online",
            page=None,
            lecture_number=None,
            topic=None,
            metadata={"title": title, "url": url, "domain": domain, "provider": provider},
        )
        results.append(RetrievedChunk(chunk=chunk, semantic_score=0.5, authority_score=0.8))
        self.last_status["accepted_results"] += 1

    def _retrieve_tavily(self, query: str, results: list[RetrievedChunk], max_results: int) -> None:
        if not TAVILY_API_KEY:
            self.last_status["providers"].append({"name": "tavily", "status": "skipped", "reason": "missing TAVILY_API_KEY"})
            return

        try:
            import requests

            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False,
                    "include_domains": APPROVED_DOMAINS,
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("results", [])
            self.last_status["providers"].append({"name": "tavily", "status": "ok", "raw_results": len(rows)})
            self.last_status["raw_results"] += len(rows)
            for row in rows:
                self._append_result(
                    results,
                    title=str(row.get("title", "")),
                    body=str(row.get("content", "")),
                    url=str(row.get("url", "")),
                    provider="tavily",
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.last_status["providers"].append({"name": "tavily", "status": "error", "reason": error})
            self.last_status["errors"].append(error)

    def _ddgs_class(self):
        try:
            from ddgs import DDGS

            return DDGS, "ddgs"
        except Exception:
            from duckduckgo_search import DDGS

            return DDGS, "duckduckgo_search"

    def _ddgs_query_variants(self, query: str) -> list[str]:
        return [
            query,
            f"{query} OpenAI documentation Hugging Face arxiv LangChain",
        ]

    def _retrieve_ddgs(self, query: str, results: list[RetrievedChunk], max_results: int) -> None:
        try:
            ddgs_class, provider_name = self._ddgs_class()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.last_status["providers"].append({"name": "ddgs", "status": "error", "reason": error})
            self.last_status["errors"].append(error)
            return

        provider_raw = 0
        try:
            seen_urls = {result.chunk.source for result in results}
            with ddgs_class() as ddgs:
                for search_query in self._ddgs_query_variants(query):
                    rows = list(ddgs.text(search_query, max_results=max_results * 2))
                    provider_raw += len(rows)
                    self.last_status["raw_results"] += len(rows)
                    for row in rows:
                        url = str(row.get("href") or row.get("url") or "")
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)
                        self._append_result(
                            results,
                            title=str(row.get("title", "")),
                            body=str(row.get("body", "") or row.get("snippet", "")),
                            url=url,
                            provider=provider_name,
                        )
                    if self.last_status["accepted_results"] >= max_results:
                        break
            self.last_status["providers"].append({"name": provider_name, "status": "ok", "raw_results": provider_raw})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.last_status["providers"].append({"name": provider_name, "status": "error", "reason": error})
            self.last_status["errors"].append(error)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self.last_status = self._empty_status()
        if not self.enabled:
            self.last_status["message"] = "Online retrieval is disabled by ENABLE_ONLINE_RAG=false."
            return []

        clean_query = _web_query(query)
        results: list[RetrievedChunk] = []
        self._retrieve_tavily(clean_query, results, max_results=max(top_k, 5))
        if len(results) < top_k:
            self._retrieve_ddgs(clean_query, results, max_results=max(top_k, 5))

        if not results:
            if self.last_status["errors"]:
                self.last_status["message"] = "Online retrieval ran but every provider failed or returned no approved results."
            elif self.last_status["raw_results"] == 0:
                self.last_status["message"] = "Online providers returned zero raw results."
            else:
                self.last_status["message"] = "Online providers returned results, but none matched the approved-domain allowlist."
            return []

        ranked = rerank(query, results, top_k=top_k)
        self.last_status["message"] = f"Online retrieval returned {len(ranked)} approved result(s)."
        return ranked
