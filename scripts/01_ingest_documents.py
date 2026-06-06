import _bootstrap  # noqa: F401

from src.config import ensure_dirs
from src.ingestion.load_documents import load_all_documents
from src.ingestion.chunk_documents import chunk_documents, save_chunks

if __name__ == "__main__":
    ensure_dirs()
    docs = load_all_documents()
    chunks = chunk_documents(docs)
    save_chunks(chunks)
    print(f"Ingested {len(docs)} document records and created {len(chunks)} chunks.")
