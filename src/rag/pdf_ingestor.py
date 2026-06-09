from collections import OrderedDict
from pathlib import Path

INGESTOR_CACHE: OrderedDict = OrderedDict()
_CACHE_CAPACITY = 10


def _evict_lru() -> None:
    while len(INGESTOR_CACHE) >= _CACHE_CAPACITY:
        INGESTOR_CACHE.popitem(last=False)


def ingest_pdf(pdf_bytes: bytes, collection_name: str = "session_pdf") -> dict:
    """
    Ingest a PDF from raw bytes into a ChromaDB collection.
    Returns a status dict with keys: collection, chunks, status, error.
    """
    if collection_name in INGESTOR_CACHE:
        INGESTOR_CACHE.move_to_end(collection_name)
        return INGESTOR_CACHE[collection_name]

    try:
        import fitz  # PyMuPDF
    except ImportError:
        return {"collection": collection_name, "chunks": 0, "status": "error", "error": "PyMuPDF not installed. Run: pip install pymupdf"}

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        return {"collection": collection_name, "chunks": 0, "status": "error", "error": f"Missing dependency: {exc}"}

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = [page.get_text() for page in doc]
        doc.close()
    except Exception as exc:
        return {"collection": collection_name, "chunks": 0, "status": "error", "error": f"PDF parse error: {exc}"}

    raw_text = "\n".join(pages_text)
    chunk_size = 400
    overlap = 80
    words = raw_text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap

    if not chunks:
        return {"collection": collection_name, "chunks": 0, "status": "empty", "error": "No text extracted from PDF"}

    try:
        from src.config import DATA_DIR
        chroma_path = str(DATA_DIR / "vector_db" / "chroma")
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=chroma_path)
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        collection = client.create_collection(collection_name)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        ids = [f"{collection_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": "uploaded_pdf", "chunk_index": i} for i in range(len(chunks))]
        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)

        result = {"collection": collection_name, "chunks": len(chunks), "status": "ok", "error": None}
        _evict_lru()
        INGESTOR_CACHE[collection_name] = result
        return result

    except Exception as exc:
        return {"collection": collection_name, "chunks": 0, "status": "error", "error": str(exc)}


def query_pdf_collection(query: str, collection_name: str = "session_pdf", top_k: int = 5) -> list[dict]:
    """Query a ChromaDB collection and return chunk dicts."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from src.config import DATA_DIR

        chroma_path = str(DATA_DIR / "vector_db" / "chroma")
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(collection_name)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode([query], show_progress_bar=False).tolist()
        results = collection.query(query_embeddings=embedding, n_results=min(top_k, collection.count()))

        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            dist = results["distances"][0][i] if results.get("distances") else 1.0
            chunks.append({
                "text": doc,
                "source": "uploaded_pdf",
                "source_type": "pdf_upload",
                "chunk_id": results["ids"][0][i],
                "page": meta.get("chunk_index"),
                "metadata": meta,
                "score": max(0.0, 1.0 - dist),
            })
        return chunks
    except Exception:
        return []


def clear_pdf_collection(collection_name: str = "session_pdf") -> None:
    """Drop the ChromaDB collection and remove from cache."""
    INGESTOR_CACHE.pop(collection_name, None)
    try:
        import chromadb
        from src.config import DATA_DIR
        chroma_path = str(DATA_DIR / "vector_db" / "chroma")
        client = chromadb.PersistentClient(path=chroma_path)
        client.delete_collection(collection_name)
    except Exception:
        pass
