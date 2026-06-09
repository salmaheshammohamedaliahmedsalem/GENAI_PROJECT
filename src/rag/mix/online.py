"""
online.py — Multi-source online retriever.

Tier 1 (no API key required):
    DuckDuckGo, Wikipedia, arXiv, Semantic Scholar, GitHub

Tier 2 (API key or optional package required):
    StackExchange, YouTube Data API + transcripts

Each provider is isolated: one failure never prevents the others from running.

Public API
----------
    MultiSourceOnlineRetriever(sources, per_source)
        .candidates(search_query) -> list[dict]
        .diagnostics()            -> dict
"""
from __future__ import annotations

import os
import re
from xml.etree import ElementTree as ET

from src.rag.mix.config import MAX_TEXT_CHARS
from src.rag.mix.http_client import http_get
from src.rag.mix.reranker import _make_record, _online_chunk


class MultiSourceOnlineRetriever:
    """
    Retrieves candidate chunks from up to 7 online sources.

    All source methods follow the same pattern:
        _src_<name>(self, q: str, out: list[dict]) -> None
    They append normalised chunk dicts to *out* and log status.
    """

    ALL_SOURCES = [
        "duckduckgo",
        "wikipedia",
        "arxiv",
        "semantic_scholar",
        "github",
        "stackexchange",
        "youtube",
    ]

    def __init__(
        self,
        sources: list[str] | None = None,
        per_source: int = 4,
    ) -> None:
        self.sources: list[str] = sources or self.ALL_SOURCES
        self.per_source: int = per_source
        self.status: list[dict] = []

    def diagnostics(self) -> dict:
        return {"providers": self.status}

    def candidates(self, search_query: str) -> list[dict]:
        """Fetch raw chunks from all enabled providers. Returns _make_record dicts."""
        self.status = []
        out: list[dict] = []
        dispatch = {
            "duckduckgo":       self._src_duckduckgo,
            "wikipedia":        self._src_wikipedia,
            "arxiv":            self._src_arxiv,
            "semantic_scholar": self._src_semantic_scholar,
            "github":           self._src_github,
            "stackexchange":    self._src_stackexchange,
            "youtube":          self._src_youtube,
        }
        for name in self.sources:
            fn = dispatch.get(name)
            if fn:
                fn(search_query, out)
            else:
                self._log(name, "unknown_source")
        return [_make_record(c) for c in out]

    def _log(self, name: str, status: str, **extra) -> None:
        self.status.append({"name": name, "status": status, **extra})

    # ---- Tier 1 -----------------------------------------------------------

    def _src_duckduckgo(self, q: str, out: list[dict]) -> None:
        try:
            try:
                from ddgs import DDGS  # noqa: PLC0415
            except Exception:
                from duckduckgo_search import DDGS  # noqa: PLC0415
        except Exception as e:
            self._log("duckduckgo", "skipped", reason=f"package missing: {e}")
            return
        try:
            n = 0
            with DDGS() as ddgs:
                for row in ddgs.text(q, max_results=self.per_source):
                    out.append(_online_chunk(
                        "duckduckgo", n,
                        row.get("title", ""),
                        row.get("body") or row.get("snippet", ""),
                        row.get("href") or row.get("url", ""),
                    ))
                    n += 1
            self._log("duckduckgo", "ok", results=n)
        except Exception as e:
            self._log("duckduckgo", "error", reason=str(e))

    def _src_wikipedia(self, q: str, out: list[dict]) -> None:
        try:
            r = http_get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "list": "search",
                    "srsearch": q, "format": "json", "srlimit": self.per_source,
                },
                headers={"User-Agent": "mix-rag/1.0"},
            )
            n = 0
            for h in r.json().get("query", {}).get("search", []):
                title = h.get("title", "")
                snip = re.sub(r"<[^>]+>", "", h.get("snippet", ""))
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                out.append(_online_chunk("wikipedia", n, title, snip, url))
                n += 1
            self._log("wikipedia", "ok", results=n)
        except Exception as e:
            self._log("wikipedia", "error", reason=str(e))

    def _src_arxiv(self, q: str, out: list[dict]) -> None:
        try:
            search = " AND ".join(q.split()[:8]) or q
            r = http_get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{search}",
                    "start": 0,
                    "max_results": self.per_source,
                },
                headers={"User-Agent": "mix-rag/1.0"},
            )
            ns = {"a": "http://www.w3.org/2005/Atom"}
            n = 0
            for entry in ET.fromstring(r.text).findall("a:entry", ns):
                title   = (entry.findtext("a:title",   "", ns) or "").strip()
                summary = (entry.findtext("a:summary", "", ns) or "").strip()
                url     = (entry.findtext("a:id",      "", ns) or "").strip()
                out.append(_online_chunk("arxiv", n, title, summary, url))
                n += 1
            self._log("arxiv", "ok", results=n)
        except Exception as e:
            self._log("arxiv", "error", reason=str(e))

    def _src_semantic_scholar(self, q: str, out: list[dict]) -> None:
        try:
            headers: dict[str, str] = {"User-Agent": "mix-rag/1.0"}
            if os.getenv("S2_API_KEY"):
                headers["x-api-key"] = os.getenv("S2_API_KEY")  # type: ignore[assignment]
            r = http_get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": q, "limit": self.per_source,
                    "fields": "title,abstract,url,year",
                },
                headers=headers,
                retries=5,
            )
            n = 0
            for p in r.json().get("data", []) or []:
                out.append(_online_chunk(
                    "semantic_scholar", n,
                    p.get("title", ""),
                    p.get("abstract") or "",
                    p.get("url") or "",
                ))
                n += 1
            self._log("semantic_scholar", "ok", results=n)
        except Exception as e:
            self._log("semantic_scholar", "error", reason=str(e))

    def _src_github(self, q: str, out: list[dict]) -> None:
        try:
            headers: dict[str, str] = {
                "User-Agent": "mix-rag/1.0",
                "Accept": "application/vnd.github+json",
            }
            if os.getenv("GITHUB_TOKEN"):
                headers["Authorization"] = f"Bearer {os.getenv('GITHUB_TOKEN')}"
            r = http_get(
                "https://api.github.com/search/repositories",
                params={"q": q, "sort": "stars", "order": "desc",
                        "per_page": self.per_source},
                headers=headers,
            )
            n = 0
            for repo in r.json().get("items", []) or []:
                body = (repo.get("description") or "") + (
                    f" (stars: {repo.get('stargazers_count', 0)}, "
                    f"lang: {repo.get('language')})"
                )
                out.append(_online_chunk(
                    "github", n, repo.get("full_name", ""),
                    body, repo.get("html_url", ""),
                ))
                n += 1
            self._log("github", "ok", results=n)
        except Exception as e:
            self._log("github", "error", reason=str(e))

    # ---- Tier 2 -----------------------------------------------------------

    def _src_stackexchange(self, q: str, out: list[dict]) -> None:
        try:
            r = http_get(
                "https://api.stackexchange.com/2.3/search/advanced",
                params={
                    "order": "desc", "sort": "relevance", "q": q,
                    "site": "stackoverflow", "filter": "withbody",
                    "pagesize": self.per_source,
                },
                headers={"User-Agent": "mix-rag/1.0"},
            )
            n = 0
            for it in r.json().get("items", []) or []:
                body = re.sub(r"<[^>]+>", " ", it.get("body", "") or "")
                out.append(_online_chunk(
                    "stackexchange", n,
                    it.get("title", ""), body, it.get("link", ""),
                ))
                n += 1
            self._log("stackexchange", "ok", results=n)
        except Exception as e:
            self._log("stackexchange", "error", reason=str(e))

    def _src_youtube(self, q: str, out: list[dict]) -> None:
        key = os.getenv("YOUTUBE_API_KEY")
        if not key:
            self._log("youtube", "skipped", reason="missing YOUTUBE_API_KEY")
            return
        try:
            r = http_get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part": "snippet", "q": q, "type": "video",
                    "maxResults": self.per_source, "key": key,
                },
            )
            n = 0
            for it in r.json().get("items", []) or []:
                vid = it.get("id", {}).get("videoId", "")
                snip = it.get("snippet", {})
                body = snip.get("description", "")
                transcript = self._youtube_transcript(vid)
                if transcript:
                    body = f"{body}\nTranscript: {transcript}"
                out.append(_online_chunk(
                    "youtube", n, snip.get("title", ""), body,
                    f"https://www.youtube.com/watch?v={vid}",
                ))
                n += 1
            self._log("youtube", "ok", results=n)
        except Exception as e:
            self._log("youtube", "error", reason=str(e))

    @staticmethod
    def _youtube_transcript(video_id: str) -> str:
        if not video_id:
            return ""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi  # noqa: PLC0415
            segs = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join(s["text"] for s in segs)[:MAX_TEXT_CHARS]
        except Exception:
            return ""
