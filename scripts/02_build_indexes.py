import _bootstrap  # noqa: F401

from src.config import ensure_dirs
from src.ingestion.build_indexes import build_indexes

if __name__ == "__main__":
    ensure_dirs()
    build_indexes()
    print("Indexes built.")
