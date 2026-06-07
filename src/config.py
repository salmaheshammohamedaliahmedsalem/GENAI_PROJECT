from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
COURSE_PDFS_DIR = RAW_DIR / "course_pdfs"
PROJECT_DOCS_DIR = RAW_DIR / "project_docs"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db" / "chroma"
FINETUNE_DIR = DATA_DIR / "finetune"
EVAL_DIR = DATA_DIR / "eval"
OUTPUTS_DIR = ROOT_DIR / "outputs"
TRACE_DIR = OUTPUTS_DIR / "traces"

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FINETUNE_BASE_MODEL = os.getenv("FINETUNE_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
ENABLE_ONLINE_RAG = os.getenv("ENABLE_ONLINE_RAG", "true").lower() == "true"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K_SEMANTIC = int(os.getenv("TOP_K_SEMANTIC", "8"))
TOP_K_KEYWORD = int(os.getenv("TOP_K_KEYWORD", "8"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

APPROVED_DOMAINS = [
    "openai.com",
    "docs.openai.com",
    "huggingface.co",
    "python.langchain.com",
    "langchain-ai.github.io",
    "arxiv.org",
    "aclanthology.org",
    "paperswithcode.com",
]

def ensure_dirs() -> None:
    for path in [
        COURSE_PDFS_DIR, PROJECT_DOCS_DIR, PROCESSED_DIR, VECTOR_DB_DIR,
        FINETUNE_DIR, EVAL_DIR, OUTPUTS_DIR, TRACE_DIR
    ]:
        path.mkdir(parents=True, exist_ok=True)
