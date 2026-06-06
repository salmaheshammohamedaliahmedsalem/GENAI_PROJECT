import json
import csv
import re
from pathlib import Path
from tqdm import tqdm

from utils import (
    EXTRACTED_TEXT_DIR,
    CHUNKS_DIR,
    METADATA_DIR,
    guess_topic,
    get_logger
)

logger = get_logger("chunk_lectures")

def split_text_into_chunks(text: str, max_words: int = 800) -> list:
    """
    Splits text into chunks of maximum words. Prefers paragraph boundaries,
    and falls back to sentence boundaries if needed.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
        
    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for p in paragraphs:
        p_words = p.split()
        if not p_words:
            continue
        if current_word_count + len(p_words) > max_words:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [p]
            current_word_count = len(p_words)
        else:
            current_chunk.append(p)
            current_word_count += len(p_words)
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    # Handle any single paragraph that exceeds max_words
    final_chunks = []
    for c in chunks:
        c_words = c.split()
        if len(c_words) <= max_words:
            final_chunks.append(c)
        else:
            # Split by sentence
            sentences = re.split(r"(?<=[.!?])\s+", c)
            curr_c = []
            curr_w = 0
            for s in sentences:
                s_words = s.split()
                if curr_w + len(s_words) > max_words:
                    if curr_c:
                        final_chunks.append(" ".join(curr_c))
                    curr_c = [s]
                    curr_w = len(s_words)
                else:
                    curr_c.append(s)
                    curr_w += len(s_words)
            if curr_c:
                final_chunks.append(" ".join(curr_c))
                
    return final_chunks

def main():
    logger.info("Starting lecture chunking pipeline...")
    
    combined_pages_file = EXTRACTED_TEXT_DIR / "all_lectures_pages.jsonl"
    
    if not combined_pages_file.exists():
        logger.error(f"Combined pages file not found at {combined_pages_file}. Please run extract_pdf_text.py first.")
        return
        
    # Read all extracted pages
    pages = []
    try:
        with open(combined_pages_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pages.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(pages)} pages from {combined_pages_file}")
    except Exception as e:
        logger.error(f"Failed to read {combined_pages_file}: {e}")
        return
        
    chunks_data = []
    summary_data = []
    
    # Process each page
    for page in tqdm(pages, desc="Chunking pages"):
        lecture_id = page["lecture_id"]
        lecture_file = page["lecture_file"]
        page_num = page["page"]
        text = page["text"]
        
        # Split text into chunks
        page_chunks = split_text_into_chunks(text, max_words=800)
        
        for idx, chunk_text in enumerate(page_chunks):
            chunk_id = f"{lecture_id}_p{page_num}_c{idx + 1}"
            word_count = len(chunk_text.split())
            topic = guess_topic(chunk_text)
            
            chunk_record = {
                "chunk_id": chunk_id,
                "lecture_id": lecture_id,
                "lecture_file": lecture_file,
                "page": page_num,
                "topic_guess": topic,
                "text": chunk_text,
                "source": "offline_lecture"
            }
            chunks_data.append(chunk_record)
            
            # Save metadata summary
            summary_record = {
                "chunk_id": chunk_id,
                "lecture_id": lecture_id,
                "page": page_num,
                "word_count": word_count,
                "topic_guess": topic
            }
            summary_data.append(summary_record)
            
    # Save lecture chunks JSONL
    chunks_file = CHUNKS_DIR / "lecture_chunks.jsonl"
    try:
        with open(chunks_file, "w", encoding="utf-8") as f:
            for chunk in chunks_data:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(chunks_data)} chunks to {chunks_file}")
    except Exception as e:
        logger.error(f"Failed to save chunks JSONL: {e}")
        
    # Save chunk summary CSV
    summary_csv_file = METADATA_DIR / "chunk_summary.csv"
    try:
        with open(summary_csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["chunk_id", "lecture_id", "page", "word_count", "topic_guess"])
            writer.writeheader()
            writer.writerows(summary_data)
        logger.info(f"Saved chunk summary CSV to {summary_csv_file}")
    except Exception as e:
        logger.error(f"Failed to save CSV summary: {e}")
        
    logger.info("Lecture chunking pipeline completed successfully.")

if __name__ == "__main__":
    main()
