import pytest
from src.rag.mix.reranker import (
    lexical_relevance,
    metadata_score,
    authority_score,
    rerank,
    _make_record,
    _online_chunk,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _chunk(text="", source_type="course_pdf", topic=None, lecture_number=None,
           domain="", chunk_id="c1"):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "source": "test",
        "source_type": source_type,
        "page": None,
        "lecture_number": lecture_number,
        "topic": topic,
        "metadata": {"domain": domain, "title": "", "url": "", "provider": ""},
    }


def _record(text="", source_type="course_pdf", keyword_score=1.0, **kw):
    return _make_record(_chunk(text=text, source_type=source_type, **kw),
                        keyword_score=keyword_score)


# ── lexical_relevance ─────────────────────────────────────────────────────────

def test_lexical_relevance_full_match():
    score = lexical_relevance(["transformer", "attention"], "attention in transformer models")
    assert score > 0.5


def test_lexical_relevance_no_match():
    score = lexical_relevance(["transformer"], "completely unrelated content")
    assert score == 0.0


def test_lexical_relevance_empty_terms_returns_zero():
    assert lexical_relevance([], "some text") == 0.0


def test_lexical_relevance_phrase_bonus():
    # The bonus (+0.2) is only observable when coverage is < 0.8, so the sum
    # does not get capped at 1.0 before versus after the bonus.
    # terms: 3 words; the third is absent from the text (coverage = 2/3 ≈ 0.667).
    # text A has "hybrid search" as a consecutive phrase → "hybrid search absent"
    # matches the full joined phrase → bonus applies.
    # text B scatters the same two matching words → no phrase → no bonus.
    terms = ["hybrid", "search", "absent"]
    # phrase = "hybrid search absent"; only A contains it as a substring
    with_phrase    = lexical_relevance(terms, "hybrid search absent overview")
    without_phrase = lexical_relevance(terms, "hybrid method search overview")
    assert with_phrase > without_phrase


def test_lexical_relevance_capped_at_one():
    score = lexical_relevance(["rag", "rag", "rag"], "rag rag rag rag")
    assert score <= 1.0


def test_lexical_relevance_title_contributes():
    terms = ["lora"]
    with_title = lexical_relevance(terms, "", title="LoRA fine-tuning")
    without_title = lexical_relevance(terms, "", title="")
    assert with_title > without_title


# ── metadata_score ───────────────────────────────────────────────────────────

def test_metadata_score_project_query_on_project_doc():
    chunk = _chunk(source_type="project_doc")
    assert metadata_score("explain the project requirements", chunk) >= 0.5


def test_metadata_score_project_query_on_course_pdf():
    chunk = _chunk(source_type="course_pdf")
    assert metadata_score("explain the project requirements", chunk) == 0.0


def test_metadata_score_lecture_number_match():
    chunk = _chunk(lecture_number=3)
    score = metadata_score("what does lecture 3 cover", chunk)
    assert score >= 0.4


def test_metadata_score_topic_overlap():
    # topic.split() tokenises on whitespace, so use a multi-word topic where
    # one token appears in the query
    chunk = _chunk(topic="rag generation")
    score = metadata_score("explain rag methods", chunk)
    assert score > 0.0


def test_metadata_score_rag_query_rag_topic():
    chunk = _chunk(
        source_type="course_pdf",
        topic="retrieval-augmented generation",
        text="retrieval augmented generation architecture overview",
    )
    score = metadata_score("explain rag", chunk)
    assert score >= 0.7


def test_metadata_score_capped_at_one():
    chunk = _chunk(source_type="project_doc", topic="retrieval-augmented generation",
                   lecture_number=1,
                   text="retrieval augmented generation architecture overview knowledge base")
    score = metadata_score("rag project lecture 1", chunk)
    assert score <= 1.0


# ── authority_score ───────────────────────────────────────────────────────────

def test_authority_score_course_pdf():
    assert authority_score(_chunk(source_type="course_pdf")) == 1.0


def test_authority_score_project_doc():
    assert authority_score(_chunk(source_type="project_doc")) == 1.0


def test_authority_score_arxiv():
    chunk = _chunk(source_type="online", domain="arxiv.org")
    assert authority_score(chunk) == 0.85


def test_authority_score_huggingface():
    chunk = _chunk(source_type="online", domain="huggingface.co")
    assert authority_score(chunk) == 0.80


def test_authority_score_unknown_online_domain():
    chunk = _chunk(source_type="online", domain="example.com")
    assert authority_score(chunk) == 0.4


def test_authority_score_unknown_source_type():
    chunk = _chunk(source_type="unknown_type")
    assert authority_score(chunk) == 0.5


# ── _make_record / _online_chunk ─────────────────────────────────────────────

def test_make_record_has_required_keys():
    rec = _make_record(_chunk())
    assert "chunk" in rec
    assert "semantic_score" in rec
    assert "keyword_score" in rec


def test_make_record_stores_scores():
    rec = _make_record(_chunk(), semantic_score=0.7, keyword_score=3.2)
    assert rec["semantic_score"] == 0.7
    assert rec["keyword_score"] == 3.2


def test_online_chunk_truncates_long_text():
    from src.rag.mix.config import MAX_TEXT_CHARS
    long_body = "word " * 500
    chunk = _online_chunk("test", 0, "Title", long_body, "https://example.com")
    assert len(chunk["text"]) <= MAX_TEXT_CHARS


def test_online_chunk_extracts_domain():
    chunk = _online_chunk("wiki", 0, "Title", "body", "https://en.wikipedia.org/wiki/RAG")
    assert chunk["metadata"]["domain"] == "en.wikipedia.org"


def test_online_chunk_source_type_is_online():
    chunk = _online_chunk("arxiv", 0, "Title", "body", "https://arxiv.org/abs/1234")
    assert chunk["source_type"] == "online"


# ── rerank ────────────────────────────────────────────────────────────────────

def test_rerank_empty_input():
    assert rerank("query", []) == []


def test_rerank_returns_at_most_top_k():
    records = [_record(text=f"document about rag number {i}", chunk_id=f"c{i}")
               for i in range(20)]
    result = rerank("rag", records, top_k=5)
    assert len(result) <= 5


def test_rerank_sorted_descending_by_final_score():
    records = [_record(text=f"text {i}", chunk_id=f"c{i}") for i in range(10)]
    result = rerank("rag", records, top_k=10)
    scores = [r["final_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_rerank_sets_final_score_on_each_record():
    records = [_record(text="retrieval augmented generation explained")]
    result = rerank("rag", records, top_k=5)
    assert result[0]["final_score"] > 0.0


def test_rerank_sets_relevance_score():
    records = [_record(text="rag retrieval")]
    result = rerank("rag", records, top_k=5, rerank_mode="lexical")
    assert result[0].get("relevance_score") is not None


def test_rerank_lexical_mode_does_not_load_cross_encoder(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        "src.rag.mix.reranker._get_cross_encoder",
        lambda model: loaded.append(model) or None,
    )
    records = [_record(text="rag explanation", chunk_id="c1")]
    rerank("rag", records, top_k=5, rerank_mode="lexical")
    assert not loaded


def test_rerank_course_pdf_outranks_unknown_online_on_same_text():
    query = "explain retrieval augmented generation"
    text = "retrieval augmented generation explanation"
    offline_rec = _record(text=text, source_type="course_pdf", chunk_id="off1")
    online_rec  = _record(text=text, source_type="online",     chunk_id="on1")
    online_rec["chunk"]["metadata"]["domain"] = "example.com"
    result = rerank(query, [online_rec, offline_rec], top_k=2, rerank_mode="lexical")
    assert result[0]["chunk"]["source_type"] == "course_pdf"
