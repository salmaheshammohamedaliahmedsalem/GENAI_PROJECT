import json
import re
from pathlib import Path
import fitz  # PyMuPDF
from tqdm import tqdm

from utils import (
    RAW_PDFS_DIR,
    EXTRACTED_TEXT_DIR,
    get_logger
)

logger = get_logger("extract_pdf_text")

def extract_lecture_id(filename: str) -> str:
    """
    Extracts a standardized lecture_id (e.g., 'lecture_7') from file names
    like 'LLM Lecture 7 (1).pdf' or 'LLM Lecture 1.pdf'.
    """
    match = re.search(r"Lecture\s+(\d+)", filename, re.IGNORECASE)
    if match:
        return f"lecture_{match.group(1)}"
    # Fallback to a sanitized version of the filename
    clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", filename.replace(".pdf", "")).lower()
    return clean_name

def normalize_text(text: str) -> str:
    """
    Normalizes whitespaces while preserving structural newlines/paragraphs.
    """
    if not text:
        return ""
    # Strip leading/trailing whitespaces per line
    lines = [line.strip() for line in text.split("\n")]
    # Remove empty lines that are consecutive (keep at most one empty line to represent paragraphs)
    cleaned_lines = []
    prev_was_empty = False
    for line in lines:
        if line == "":
            if not prev_was_empty:
                cleaned_lines.append(line)
                prev_was_empty = True
        else:
            cleaned_lines.append(line)
            prev_was_empty = False
            
    normalized = "\n".join(cleaned_lines).strip()
    # Replace multiple inline spaces/tabs with single space
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized

def extract_text_from_pdf(pdf_path: Path) -> list:
    """
    Extracts text from a single PDF and returns a list of dictionaries containing page metadata and text.
    """
    lecture_file = pdf_path.name
    lecture_id = extract_lecture_id(lecture_file)
    logger.info(f"Extracting text from {lecture_file} (mapped to ID: {lecture_id})")
    
    pages_data = []
    
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        
        # Iterate through pages using tqdm
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            raw_text = page.get_text("text")
            clean_text = normalize_text(raw_text)
            
            # Skip only if completely empty after cleaning
            if not clean_text:
                continue
                
            pages_data.append({
                "lecture_file": lecture_file,
                "lecture_id": lecture_id,
                "page": page_idx + 1,  # 1-indexed page numbers
                "text": clean_text,
                "char_count": len(clean_text)
            })
            
        doc.close()
        logger.info(f"Successfully extracted {len(pages_data)} pages from {lecture_file}")
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        
    return pages_data

def main():
    logger.info("Starting PDF text extraction pipeline...")
    
    # List all raw PDF files
    pdf_paths = sorted(list(RAW_PDFS_DIR.glob("*.pdf")))
    
    if not pdf_paths:
        logger.warning(f"No PDF files found in {RAW_PDFS_DIR}. Please add them.")
        return
        
    logger.info(f"Found {len(pdf_paths)} raw PDFs to process.")
    
    all_lectures_data = []
    
    for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
        lecture_pages = extract_text_from_pdf(pdf_path)
        if not lecture_pages:
            continue
            
        # Save individual JSONL file
        lecture_id = lecture_pages[0]["lecture_id"]
        individual_file = EXTRACTED_TEXT_DIR / f"{lecture_id}_pages.jsonl"
        
        try:
            with open(individual_file, "w", encoding="utf-8") as f:
                for page in lecture_pages:
                    f.write(json.dumps(page, ensure_ascii=False) + "\n")
            logger.info(f"Saved individual JSONL to {individual_file}")
        except Exception as e:
            logger.error(f"Failed to write individual file for {lecture_id}: {e}")
            
        all_lectures_data.extend(lecture_pages)
        
    # Save combined JSONL file
    combined_file = EXTRACTED_TEXT_DIR / "all_lectures_pages.jsonl"
    try:
        with open(combined_file, "w", encoding="utf-8") as f:
            for page in all_lectures_data:
                f.write(json.dumps(page, ensure_ascii=False) + "\n")
        logger.info(f"Successfully saved combined JSONL with {len(all_lectures_data)} pages to {combined_file}")
    except Exception as e:
        logger.error(f"Failed to write combined file: {e}")
        
    logger.info("PDF text extraction completed successfully.")

if __name__ == "__main__":
    main()
