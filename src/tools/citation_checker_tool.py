from src.rag.citations import validate_citations

def check_citations(answer: str, retrieved_chunks: list) -> dict:
    return validate_citations(answer, retrieved_chunks)