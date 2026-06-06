from src.rag.offline_retriever import OfflineRetriever

def test_offline_retriever_constructs():
    retriever = OfflineRetriever()
    assert retriever is not None