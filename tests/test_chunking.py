from src.ingestion.chunk_documents import split_text, chunk_documents

def test_split_text():
    chunks = split_text("a" * 2000, size=500, overlap=50)
    assert len(chunks) > 1

def test_chunking_preserves_metadata():
    docs = [{"text": "hello world", "source": "LLM Lecture 1.pdf", "source_type": "course_pdf", "page": 1, "lecture_number": 1, "topic": "Intro", "metadata": {}}]
    chunks = chunk_documents(docs)
    assert chunks[0].source == "LLM Lecture 1.pdf"
    assert chunks[0].page == 1