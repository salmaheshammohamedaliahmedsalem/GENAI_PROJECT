from pathlib import Path
from typing import Any
from pypdf import PdfReader
from docx import Document
from src.config import COURSE_PDFS_DIR, PROJECT_DOCS_DIR
from src.utils.text_utils import clean_text, infer_lecture_number, infer_topic

def load_pdf(path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(path))
    records = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            records.append({
                "text": text,
                "source": path.name,
                "source_type": "course_pdf",
                "page": i,
                "lecture_number": infer_lecture_number(path.name),
                "topic": infer_topic(path.name, text),
                "metadata": {"path": str(path)},
            })
    return records

def load_docx(path: Path) -> list[dict[str, Any]]:
    doc = Document(str(path))
    text = clean_text("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
    if not text:
        return []
    return [{
        "text": text,
        "source": path.name,
        "source_type": "project_doc",
        "page": None,
        "lecture_number": None,
        "topic": infer_topic(path.name, text),
        "metadata": {"path": str(path)},
    }]

def load_text_file(path: Path) -> list[dict[str, Any]]:
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    return [{
        "text": text,
        "source": path.name,
        "source_type": "project_doc",
        "page": None,
        "lecture_number": None,
        "topic": infer_topic(path.name, text),
        "metadata": {"path": str(path)},
    }] if text else []

def load_all_documents() -> list[dict[str, Any]]:
    records = []
    for path in COURSE_PDFS_DIR.glob("*.pdf"):
        records.extend(load_pdf(path))
    for path in PROJECT_DOCS_DIR.glob("*"):
        if path.suffix.lower() == ".docx":
            records.extend(load_docx(path))
        elif path.suffix.lower() in {".txt", ".md"}:
            records.extend(load_text_file(path))
    return records