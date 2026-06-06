from src.schemas import DocumentChunk, RetrievedChunk
from src.rag.citations import citation_label, validate_citations

def test_citation_validation():
    chunk = DocumentChunk(chunk_id="c1", text="RAG text", source="lecture.pdf", source_type="course_pdf", page=1)
    r = RetrievedChunk(chunk=chunk)
    label = citation_label(r)
    result = validate_citations(f"Answer {label}", [r])
    assert result["valid"]