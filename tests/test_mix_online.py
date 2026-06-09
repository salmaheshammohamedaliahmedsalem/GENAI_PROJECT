import pytest
from src.rag.mix.online import MultiSourceOnlineRetriever


# ── construction ─────────────────────────────────────────────────────────────

def test_constructs_with_defaults():
    retr = MultiSourceOnlineRetriever()
    assert retr.sources == MultiSourceOnlineRetriever.ALL_SOURCES
    assert retr.per_source == 4


def test_constructs_with_source_subset():
    retr = MultiSourceOnlineRetriever(sources=["wikipedia", "arxiv"])
    assert retr.sources == ["wikipedia", "arxiv"]


def test_diagnostics_initially_empty():
    retr = MultiSourceOnlineRetriever()
    assert retr.diagnostics() == {"providers": []}


# ── record format ─────────────────────────────────────────────────────────────

def _fake_ddgs_class(results):
    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, q, max_results):
            return results
    return FakeDDGS


def test_candidates_returns_list_of_records(monkeypatch):
    FakeDDGS = _fake_ddgs_class([
        {"title": "RAG paper", "body": "About retrieval", "href": "https://arxiv.org/abs/1"}
    ])
    monkeypatch.setattr("src.rag.mix.online.DDGS", FakeDDGS, raising=False)
    retr = MultiSourceOnlineRetriever(sources=["duckduckgo"], per_source=1)

    def patched_ddg(self, q, out):
        out.append({
            "chunk_id": "online_duckduckgo_0",
            "text": "About retrieval",
            "source": "https://arxiv.org/abs/1",
            "source_type": "online",
            "page": None,
            "lecture_number": None,
            "topic": None,
            "metadata": {"title": "RAG paper", "url": "https://arxiv.org/abs/1",
                         "domain": "arxiv.org", "provider": "duckduckgo"},
        })

    monkeypatch.setattr(MultiSourceOnlineRetriever, "_src_duckduckgo", patched_ddg)
    records = retr.candidates("rag")

    assert isinstance(records, list)
    for r in records:
        assert "chunk" in r
        assert "chunk_id" in r["chunk"]
        assert "text" in r["chunk"]
        assert r["chunk"]["source_type"] == "online"


# ── DuckDuckGo ────────────────────────────────────────────────────────────────

def test_duckduckgo_ok(monkeypatch):
    def fake_src(self, q, out):
        out.append({
            "chunk_id": "online_duckduckgo_0",
            "text": "rag retrieval explanation",
            "source": "https://example.com",
            "source_type": "online",
            "page": None, "lecture_number": None, "topic": None,
            "metadata": {"title": "t", "url": "https://example.com",
                         "domain": "example.com", "provider": "duckduckgo"},
        })
        self._log("duckduckgo", "ok", results=1)

    monkeypatch.setattr(MultiSourceOnlineRetriever, "_src_duckduckgo", fake_src)
    retr = MultiSourceOnlineRetriever(sources=["duckduckgo"])
    records = retr.candidates("rag")
    assert len(records) == 1
    assert any(p["name"] == "duckduckgo" and p["status"] == "ok"
               for p in retr.diagnostics()["providers"])


def test_duckduckgo_skipped_on_missing_package(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name in ("ddgs", "duckduckgo_search"):
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    retr = MultiSourceOnlineRetriever(sources=["duckduckgo"])
    records = retr.candidates("rag")
    assert records == []
    assert any(p["status"] == "skipped" for p in retr.diagnostics()["providers"])


# ── Wikipedia ────────────────────────────────────────────────────────────────

def test_wikipedia_ok(monkeypatch):
    import types

    fake_resp = types.SimpleNamespace(
        json=lambda: {"query": {"search": [
            {"title": "RAG", "snippet": "retrieval augmented generation explanation"}
        ]}}
    )
    monkeypatch.setattr("src.rag.mix.online.http_get", lambda *a, **kw: fake_resp)

    retr = MultiSourceOnlineRetriever(sources=["wikipedia"], per_source=1)
    records = retr.candidates("rag")
    assert len(records) == 1
    assert records[0]["chunk"]["source_type"] == "online"
    assert any(p["name"] == "wikipedia" and p["status"] == "ok"
               for p in retr.diagnostics()["providers"])


def test_wikipedia_error_logged(monkeypatch):
    monkeypatch.setattr("src.rag.mix.online.http_get",
                        lambda *a, **kw: (_ for _ in ()).throw(Exception("timeout")))
    retr = MultiSourceOnlineRetriever(sources=["wikipedia"])
    records = retr.candidates("rag")
    assert records == []
    assert any(p["name"] == "wikipedia" and p["status"] == "error"
               for p in retr.diagnostics()["providers"])


# ── arXiv ─────────────────────────────────────────────────────────────────────

def test_arxiv_ok(monkeypatch):
    import types
    xml = (
        '<?xml version="1.0"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        '<entry>'
        '<title>Attention Is All You Need</title>'
        '<summary>Transformer architecture paper.</summary>'
        '<id>https://arxiv.org/abs/1706.03762</id>'
        '</entry>'
        '</feed>'
    )
    fake_resp = types.SimpleNamespace(text=xml)
    monkeypatch.setattr("src.rag.mix.online.http_get", lambda *a, **kw: fake_resp)

    retr = MultiSourceOnlineRetriever(sources=["arxiv"], per_source=1)
    records = retr.candidates("transformer")
    assert len(records) == 1
    assert "Transformer" in records[0]["chunk"]["text"] or "attention" in records[0]["chunk"]["text"].lower()


# ── provider isolation ────────────────────────────────────────────────────────

def test_one_failing_provider_does_not_block_others(monkeypatch):
    import types

    fake_resp = types.SimpleNamespace(
        json=lambda: {"query": {"search": [
            {"title": "RAG", "snippet": "retrieval augmented generation"}
        ]}}
    )

    call_count = {"n": 0}

    def side_effect(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("first provider down")
        return fake_resp

    monkeypatch.setattr("src.rag.mix.online.http_get", side_effect)

    retr = MultiSourceOnlineRetriever(sources=["arxiv", "wikipedia"], per_source=1)
    records = retr.candidates("rag")
    # At least wikipedia should succeed despite arxiv failing
    statuses = {p["name"]: p["status"] for p in retr.diagnostics()["providers"]}
    assert "arxiv" in statuses
    assert "wikipedia" in statuses


def test_youtube_skipped_without_api_key(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    retr = MultiSourceOnlineRetriever(sources=["youtube"])
    retr.candidates("rag")
    assert any(p["name"] == "youtube" and p["status"] == "skipped"
               for p in retr.diagnostics()["providers"])


# ── source subset ─────────────────────────────────────────────────────────────

def test_unknown_source_is_logged(monkeypatch):
    retr = MultiSourceOnlineRetriever(sources=["nonexistent_source"])
    retr.candidates("rag")
    assert any(p["name"] == "nonexistent_source"
               for p in retr.diagnostics()["providers"])


def test_only_requested_sources_are_queried(monkeypatch):
    called = []

    def track(self, q, out):
        called.append("wikipedia")
        self._log("wikipedia", "ok", results=0)

    monkeypatch.setattr(MultiSourceOnlineRetriever, "_src_wikipedia", track)
    retr = MultiSourceOnlineRetriever(sources=["wikipedia"])
    retr.candidates("rag")
    assert called == ["wikipedia"]
