from src.config import CHUNK_SIZE, CHUNK_OVERLAP, PROCESSED_DIR
from src.schemas import DocumentChunk
from src.utils.jsonl_utils import write_jsonl

def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start = max(start + 1, end - overlap)
    return [c for c in chunks if c]

def chunk_documents(records: list[dict]) -> list[DocumentChunk]:
    chunks = []
    for doc_idx, record in enumerate(records):
        parts = split_text(record["text"])
        for part_idx, part in enumerate(parts):
            chunk_id = f"{record['source'].replace(' ', '_')}_{record.get('page') or 0}_{part_idx}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=part,
                source=record["source"],
                source_type=record["source_type"],
                page=record.get("page"),
                lecture_number=record.get("lecture_number"),
                topic=record.get("topic"),
                metadata=record.get("metadata", {}) | {"doc_index": doc_idx, "part_index": part_idx},
            ))
    return chunks

def save_chunks(chunks: list[DocumentChunk]) -> None:
    write_jsonl(PROCESSED_DIR / "chunks.jsonl", [c.model_dump() for c in chunks])