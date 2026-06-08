import os
from src.config import VECTOR_DB_DIR, PROCESSED_DIR, EMBEDDING_MODEL, TOP_K_SEMANTIC, TOP_K_KEYWORD, TOP_K_FINAL
from src.schemas import DocumentChunk, RetrievedChunk
from src.utils.file_utils import read_pickle
from src.rag.reranker import rerank


def _load_chroma_dependencies():
    import chromadb
    from chromadb.utils import embedding_functions

    return chromadb, embedding_functions

class OfflineRetriever:
    def __init__(self):
        self.client = None
        self.emb = None
        self.collection = None
        self.bm25_data = None
        self.semantic_error = ""
        bm25_path = PROCESSED_DIR / "bm25_index.pkl"
        if bm25_path.exists():
            self.bm25_data = read_pickle(bm25_path)
        if os.getenv("ENABLE_SEMANTIC_RAG", "false").lower() == "true":
            try:
                chromadb, embedding_functions = _load_chroma_dependencies()
                self.client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
                self.emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
                self.collection = self.client.get_or_create_collection("course_chunks", embedding_function=self.emb)
            except Exception as exc:
                self.semantic_error = f"{type(exc).__name__}: {exc}"
                self.collection = None

    def retrieve(self, query: str, top_k: int = TOP_K_FINAL) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}

        try:
            if self.collection is None or self.emb is None:
                raise RuntimeError("Semantic embeddings unavailable.")
            result = self.collection.query(query_texts=[query], n_results=TOP_K_SEMANTIC)
            ids = result.get("ids", [[]])[0]
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0] if result.get("distances") else [0] * len(ids)
            for cid, doc, meta, dist in zip(ids, docs, metas, distances):
                chunk = DocumentChunk(
                    chunk_id=cid,
                    text=doc,
                    source=meta.get("source", ""),
                    source_type=meta.get("source_type", "course_pdf"),
                    page=None if meta.get("page", -1) == -1 else meta.get("page"),
                    lecture_number=None if meta.get("lecture_number", -1) == -1 else meta.get("lecture_number"),
                    topic=meta.get("topic") or None,
                    metadata=meta,
                )
                merged[cid] = RetrievedChunk(chunk=chunk, semantic_score=max(0.0, 1.0 - float(dist)))
        except Exception:
            pass

        if self.bm25_data:
            bm25 = self.bm25_data["bm25"]
            chunks = [DocumentChunk(**c) for c in self.bm25_data["chunks"]]
            scores = bm25.get_scores(query.lower().split())
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K_KEYWORD]
            for i in top_indices:
                chunk = chunks[i]
                if chunk.chunk_id not in merged:
                    merged[chunk.chunk_id] = RetrievedChunk(chunk=chunk)
                merged[chunk.chunk_id].keyword_score = float(scores[i])

        return rerank(query, list(merged.values()), top_k=top_k)
