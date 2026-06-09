from src.rag.mix.query import clean_query, expand_query, enriched_query


# ── clean_query ──────────────────────────────────────────────────────────────

def test_clean_query_strips_filler_prefix():
    assert clean_query("explain RAG") == "rag"


def test_clean_query_strips_multi_word_filler():
    assert clean_query("tell me about transformers") == "transformers"


def test_clean_query_strips_course_filler():
    result = clean_query("based on our course lectures explain LoRA")
    assert "lora" in result
    assert "course" not in result
    assert "lectures" not in result


def test_clean_query_removes_stopwords():
    result = clean_query("what is the attention mechanism in a transformer")
    tokens = result.split()
    assert "what" not in tokens
    assert "the" not in tokens
    assert "in" not in tokens
    assert "a" not in tokens
    assert "attention" in tokens
    assert "transformer" in tokens


def test_clean_query_deduplicates_tokens():
    result = clean_query("rag rag retrieval retrieval")
    tokens = result.split()
    assert len(tokens) == len(set(tokens))


def test_clean_query_lowercases():
    assert clean_query("RAG LoRA PEFT") == clean_query("rag lora peft")


def test_clean_query_falls_back_when_all_tokens_stripped():
    # Only stopwords → returns original rather than empty string
    result = clean_query("is it the")
    assert result  # non-empty


def test_clean_query_handles_empty_string():
    result = clean_query("")
    assert isinstance(result, str)


# ── expand_query ─────────────────────────────────────────────────────────────

def test_expand_query_rag():
    expansions = expand_query("rag")
    assert any("retrieval" in e for e in expansions)


def test_expand_query_lora():
    expansions = expand_query("lora")
    assert any("low-rank" in e for e in expansions)


def test_expand_query_peft():
    expansions = expand_query("peft")
    assert any("parameter efficient" in e for e in expansions)


def test_expand_query_unknown_term_returns_empty():
    assert expand_query("zebra") == []


def test_expand_query_multiple_terms_combines():
    expansions = expand_query("rag lora")
    assert len(expansions) >= 2


def test_expand_query_no_duplicates():
    expansions = expand_query("embedding embeddings")
    assert len(expansions) == len(set(expansions))


# ── enriched_query ───────────────────────────────────────────────────────────

def test_enriched_query_returns_tuple():
    result = enriched_query("explain RAG")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_enriched_query_search_string_is_str():
    search, terms = enriched_query("explain RAG")
    assert isinstance(search, str)
    assert isinstance(terms, list)


def test_enriched_query_expands_rag():
    search, _ = enriched_query("explain RAG based on our course lectures")
    assert "retrieval" in search


def test_enriched_query_terms_are_cleaned_tokens():
    _, terms = enriched_query("what is the attention mechanism")
    assert "what" not in terms
    assert "is" not in terms
    assert "attention" in terms


def test_enriched_query_no_clean_passthrough():
    search, _ = enriched_query("explain RAG", clean=False, expand=False)
    assert "explain" in search.lower()


def test_enriched_query_no_expand_skips_synonyms():
    search, _ = enriched_query("rag", expand=False)
    assert "retrieval augmented generation" not in search


def test_enriched_query_result_is_non_empty():
    search, terms = enriched_query("hybrid search in RAG")
    assert search
    assert terms
