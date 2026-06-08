from src.rag.online_retriever import OnlineRetriever


class FakeDDGS:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def text(self, query, max_results):
        return [
            {
                "title": "Models | OpenAI API",
                "href": "https://developers.openai.com/api/docs/models",
                "body": "OpenAI model documentation for API usage.",
            },
            {
                "title": "Unapproved blog",
                "href": "https://example.com/blog",
                "body": "This result should be filtered out.",
            },
        ]


def test_online_retriever_approves_subdomains():
    retriever = OnlineRetriever()

    assert retriever._approved("https://developers.openai.com/api/docs/models")
    assert retriever._approved("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct")
    assert not retriever._approved("https://example.com/not-approved")


def test_online_retriever_uses_ddgs_and_filters_allowlist(monkeypatch):
    retriever = OnlineRetriever()
    monkeypatch.setattr(retriever, "_ddgs_class", lambda: (FakeDDGS, "ddgs"))
    monkeypatch.setattr(retriever, "_retrieve_tavily", lambda query, results, max_results: None)

    results = retriever.retrieve("OpenAI API model docs", top_k=3)

    assert len(results) == 1
    assert results[0].chunk.source == "https://developers.openai.com/api/docs/models"
    assert retriever.diagnostics()["accepted_results"] == 1
    assert retriever.diagnostics()["filtered_results"] >= 1
