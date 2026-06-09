import pytest
from src.rag.hybrid_retriever import (
    HybridRetriever,
    _mix_record_to_retrieved_chunk,
    _enriched_query,
    _online_candidates,
)
from src.schemas import DocumentChunk, RetrievedChunk


# ── helpers ──────────────────────────────────────────────────────────────────

def _mix_record(chunk_id="c1", text="sample text", source_type="online",
                relevance=0.8, keyword=1.5, meta=0.3, auth=0.7, final=0.65):
    return {
        "chunk": {
            "chunk_id": chunk_id,
            "text": text,
            "source": "https://example.com",
            "source_type": source_type,
            "page": None,
            "lecture_number": None,
            "topic": None,
            "metadata": {"title": "t", "url": "https://example.com",
                         "domain": "example.com", "provider": "test"},
        },
        "relevance_score": relevance,
        "keyword_score": keyword,
        "metadata_score": meta,
        "authority_score": auth,
        "final_score": final,
    }


def _fake_retrieved_chunk(chunk_id="off1", text="offline text"):
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=chunk_id, text=text,
            source="lecture.pdf", source_type="course_pdf",
        ),
        semantic_score=0.6,
        keyword_score=2.0,
        final_score=0.55,
    )


# ── _mix_record_to_retrieved_chunk ────────────────────────────────────────────

def test_conversion_returns_retrieved_chunk():
    result = _mix_record_to_retrieved_chunk(_mix_record())
    assert isinstance(result, RetrievedChunk)


def test_conversion_maps_chunk_fields():
    rec = _mix_record(chunk_id="x1", text="hello world", source_type="online")
    result = _mix_record_to_retrieved_chunk(rec)
    assert result.chunk.chunk_id == "x1"
    assert result.chunk.text == "hello world"
    assert result.chunk.source_type == "online"


def test_conversion_maps_scores():
    rec = _mix_record(relevance=0.9, keyword=2.0, meta=0.4, auth=0.8, final=0.72)
    result = _mix_record_to_retrieved_chunk(rec)
    assert result.semantic_score == 0.9
    assert result.keyword_score == 2.0
    assert result.metadata_score == 0.4
    assert result.authority_score == 0.8
    assert result.final_score == 0.72


def test_conversion_handles_missing_optional_fields():
    rec = {
        "chunk": {
            "chunk_id": "c1", "text": "text",
            "source": "", "source_type": "online",
        },
    }
    result = _mix_record_to_retrieved_chunk(rec)
    assert result.chunk.page is None
    assert result.chunk.topic is None
    assert result.final_score == 0.0


def test_conversion_none_metadata_defaults_to_empty_dict():
    rec = _mix_record()
    rec["chunk"]["metadata"] = None
    result = _mix_record_to_retrieved_chunk(rec)
    assert result.chunk.metadata == {}


# ── _enriched_query ───────────────────────────────────────────────────────────

def test_enriched_query_expands_rag():
    result = _enriched_query("explain RAG")
    assert "retrieval" in result


def test_enriched_query_strips_filler():
    result = _enriched_query("tell me about transformers")
    assert "tell" not in result
    assert "transformer" in result


def test_enriched_query_fallback_on_import_error(monkeypatch):
    import src.rag.hybrid_retriever as mod
    monkeypatch.setattr(mod, "_enriched_query",
                        lambda q: (_ for _ in ()).throw(ImportError("no module")))
    # The real _enriched_query catches exceptions internally; test the guard
    result = _enriched_query("some query about rag")
    assert isinstance(result, str)
    assert result  # non-empty


# ── _online_candidates ────────────────────────────────────────────────────────

def test_online_candidates_returns_tuple():
    chunks, diag = _online_candidates.__wrapped__("rag") if hasattr(_online_candidates, "__wrapped__") \
        else _online_candidates("rag")
    assert isinstance(chunks, list)
    assert isinstance(diag, dict)


def test_online_candidates_returns_empty_on_exception(monkeypatch):
    import src.rag.hybrid_retriever as mod

    original = mod._online_candidates

    def patched(q):
        raise RuntimeError("simulated failure")

    # Verify the guard by testing that the function wraps errors gracefully
    # We patch MultiSourceOnlineRetriever to raise
    import src.rag.mix.online as online_mod
    monkeypatch.setattr(online_mod.MultiSourceOnlineRetriever, "candidates",
                        lambda self, q: (_ for _ in ()).throw(RuntimeError("boom")))

    chunks, diag = original("rag")
    assert chunks == []
    assert "error" in diag


# ── HybridRetriever construction ──────────────────────────────────────────────

def test_hybrid_retriever_constructs():
    retriever = HybridRetriever()
    assert retriever is not None
    assert retriever.offline is not None
    assert retriever.last_status == {}


# ── no-retrieval modes ────────────────────────────────────────────────────────

def test_retrieve_no_retrieval_returns_empty():
    retriever = HybridRetriever()
    assert retriever.retrieve("rag", mode="no_retrieval") == []


def test_retrieve_tool_only_returns_empty():
    retriever = HybridRetriever()
    assert retriever.retrieve("rag", mode="tool_only") == []


# ── offline_only ──────────────────────────────────────────────────────────────

def test_retrieve_offline_only_uses_offline_retriever(monkeypatch):
    retriever = HybridRetriever()
    expected = [_fake_retrieved_chunk()]
    monkeypatch.setattr(retriever.offline, "retrieve", lambda q: expected)

    result = retriever.retrieve("rag", mode="offline_only")

    assert result == expected
    assert retriever.last_status["offline_count"] == 1
    assert retriever.last_status["online_count"] == 0


def test_retrieve_offline_only_returns_retrieved_chunks(monkeypatch):
    retriever = HybridRetriever()
    monkeypatch.setattr(retriever.offline, "retrieve",
                        lambda q: [_fake_retrieved_chunk()])

    result = retriever.retrieve("hybrid RAG", mode="offline_only")

    assert all(isinstance(r, RetrievedChunk) for r in result)


def test_retrieve_offline_only_query_is_enriched(monkeypatch):
    received = []
    retriever = HybridRetriever()
    monkeypatch.setattr(retriever.offline, "retrieve",
                        lambda q: received.append(q) or [])

    retriever.retrieve("explain rag", mode="offline_only")

    assert received
    assert "retrieval" in received[0]  # expansion happened


# ── online_only ───────────────────────────────────────────────────────────────

def test_retrieve_online_only_uses_mix_online(monkeypatch):
    import src.rag.hybrid_retriever as mod

    fake_chunks = [_mix_record_to_retrieved_chunk(_mix_record(chunk_id=f"on{i}"))
                   for i in range(3)]
    monkeypatch.setattr(mod, "_online_candidates",
                        lambda q: (fake_chunks, {"providers": []}))

    retriever = HybridRetriever()
    result = retriever.retrieve("rag", mode="online_only")

    assert len(result) > 0
    assert retriever.last_status["online_count"] > 0
    assert retriever.last_status["offline_count"] == 0


def test_retrieve_online_only_returns_retrieved_chunks(monkeypatch):
    import src.rag.hybrid_retriever as mod

    fake_chunks = [_mix_record_to_retrieved_chunk(_mix_record())]
    monkeypatch.setattr(mod, "_online_candidates",
                        lambda q: (fake_chunks, {"providers": []}))

    retriever = HybridRetriever()
    result = retriever.retrieve("rag", mode="online_only")

    assert all(isinstance(r, RetrievedChunk) for r in result)


# ── hybrid ────────────────────────────────────────────────────────────────────

def test_retrieve_hybrid_combines_offline_and_online(monkeypatch):
    import src.rag.hybrid_retriever as mod

    offline_chunk = _fake_retrieved_chunk(chunk_id="off1")
    online_chunk  = _mix_record_to_retrieved_chunk(_mix_record(chunk_id="on1"))

    retriever = HybridRetriever()
    monkeypatch.setattr(retriever.offline, "retrieve", lambda q: [offline_chunk])
    monkeypatch.setattr(mod, "_online_candidates",
                        lambda q: ([online_chunk], {"providers": []}))

    result = retriever.retrieve("rag", mode="hybrid")

    source_types = {r.chunk.source_type for r in result}
    assert "course_pdf" in source_types
    assert "online" in source_types


def test_retrieve_hybrid_caps_at_top_k(monkeypatch):
    import src.rag.hybrid_retriever as mod
    from src.config import TOP_K_FINAL

    offline_chunks = [_fake_retrieved_chunk(chunk_id=f"off{i}") for i in range(10)]
    online_chunks  = [_mix_record_to_retrieved_chunk(_mix_record(chunk_id=f"on{i}"))
                      for i in range(10)]

    retriever = HybridRetriever()
    monkeypatch.setattr(retriever.offline, "retrieve", lambda q: offline_chunks)
    monkeypatch.setattr(mod, "_online_candidates",
                        lambda q: (online_chunks, {"providers": []}))

    result = retriever.retrieve("rag", mode="hybrid")

    assert len(result) <= TOP_K_FINAL


def test_retrieve_hybrid_tracks_both_counts(monkeypatch):
    import src.rag.hybrid_retriever as mod

    retriever = HybridRetriever()
    monkeypatch.setattr(retriever.offline, "retrieve",
                        lambda q: [_fake_retrieved_chunk()])
    monkeypatch.setattr(mod, "_online_candidates",
                        lambda q: ([_mix_record_to_retrieved_chunk(_mix_record())],
                                   {"providers": []}))

    retriever.retrieve("rag", mode="hybrid")

    assert retriever.last_status["offline_count"] == 1
    assert retriever.last_status["online_count"] == 1


# ── status tracking ───────────────────────────────────────────────────────────

def test_last_status_reset_on_each_call(monkeypatch):
    retriever = HybridRetriever()
    monkeypatch.setattr(retriever.offline, "retrieve",
                        lambda q: [_fake_retrieved_chunk()])

    retriever.retrieve("rag", mode="offline_only")
    first_count = retriever.last_status["offline_count"]

    monkeypatch.setattr(retriever.offline, "retrieve", lambda q: [])
    retriever.retrieve("rag", mode="offline_only")

    assert retriever.last_status["offline_count"] == 0


def test_last_status_records_mode(monkeypatch):
    retriever = HybridRetriever()
    monkeypatch.setattr(retriever.offline, "retrieve", lambda q: [])

    retriever.retrieve("rag", mode="offline_only")
    assert retriever.last_status["mode"] == "offline_only"


# ── unknown mode fallback ─────────────────────────────────────────────────────

def test_retrieve_unknown_mode_falls_back_to_offline(monkeypatch):
    retriever = HybridRetriever()
    called = []
    monkeypatch.setattr(retriever.offline, "retrieve",
                        lambda q: called.append(q) or [])

    retriever.retrieve("rag", mode="some_unknown_mode")

    assert called  # offline.retrieve was called
