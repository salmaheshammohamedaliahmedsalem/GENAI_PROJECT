import os
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from src.config import DATA_DIR, PROCESSED_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL
from src.schemas import DocumentChunk
from src.utils.jsonl_utils import read_jsonl
from src.utils.file_utils import write_pickle, write_json
from src.ingestion.build_metadata import build_metadata

def load_chunks() -> list[DocumentChunk]:
    rows = read_jsonl(PROCESSED_DIR / "chunks.jsonl")
    if not rows:
        rows = read_jsonl(DATA_DIR / "chunks" / "lecture_chunks.jsonl")
    return [_normalize_chunk(row) for row in rows]

def _normalize_chunk(row: dict) -> DocumentChunk:
    if "source_type" in row:
        return DocumentChunk(**row)
    metadata = {
        "lecture_id": row.get("lecture_id"),
        "raw_source": row.get("source"),
    }
    return DocumentChunk(
        chunk_id=row["chunk_id"],
        text=row["text"],
        source=row.get("lecture_file", row.get("source", "lecture")),
        source_type="course_pdf",
        page=row.get("page"),
        lecture_number=_lecture_number(row.get("lecture_id", "")),
        topic=row.get("topic_guess"),
        metadata=metadata,
    )

def _lecture_number(lecture_id: str) -> int | None:
    digits = "".join(ch for ch in lecture_id if ch.isdigit())
    return int(digits) if digits else None

def tokenize(text: str) -> list[str]:
    return text.lower().split()

def build_indexes() -> None:
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError("No chunks found. Run scripts/01_ingest_documents.py first.")

    if os.getenv("ENABLE_SEMANTIC_RAG", "false").lower() == "true":
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        collection = client.get_or_create_collection("course_chunks", embedding_function=emb)

        try:
            existing = collection.get()
            if existing.get("ids"):
                collection.delete(ids=existing["ids"])
        except Exception:
            pass

        collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{
                "source": c.source,
                "source_type": c.source_type,
                "page": c.page or -1,
                "lecture_number": c.lecture_number or -1,
                "topic": c.topic or "",
            } for c in chunks],
        )

    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
    write_pickle(PROCESSED_DIR / "bm25_index.pkl", {"bm25": bm25, "chunks": [c.model_dump() for c in chunks]})
    write_json(PROCESSED_DIR / "documents_metadata.json", build_metadata(chunks))
