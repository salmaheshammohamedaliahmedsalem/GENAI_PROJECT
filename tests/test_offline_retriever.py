from src.rag.offline_retriever import OfflineRetriever

def test_offline_retriever_constructs():
    retriever = OfflineRetriever()
    assert retriever is not None


def test_offline_retriever_does_not_require_chroma_when_semantic_disabled(monkeypatch):
    import src.rag.offline_retriever as offline_module

    monkeypatch.setenv("ENABLE_SEMANTIC_RAG", "false")
    monkeypatch.setattr(
        offline_module,
        "_load_chroma_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("Chroma should not be loaded")),
    )

    retriever = OfflineRetriever()

    assert retriever.collection is None


def test_offline_retriever_handles_chroma_import_failure(monkeypatch):
    import src.rag.offline_retriever as offline_module

    monkeypatch.setenv("ENABLE_SEMANTIC_RAG", "true")
    monkeypatch.setattr(
        offline_module,
        "_load_chroma_dependencies",
        lambda: (_ for _ in ()).throw(TypeError("protobuf compatibility failure")),
    )

    retriever = OfflineRetriever()

    assert retriever.collection is None
    assert "protobuf compatibility failure" in retriever.semantic_error
