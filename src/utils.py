import logging
from pathlib import Path
import re

# 1. Project Directory Configuration
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
EXTRACTED_TEXT_DIR = DATA_DIR / "extracted_text"
CHUNKS_DIR = DATA_DIR / "chunks"
FINETUNING_DIR = DATA_DIR / "finetuning"
METADATA_DIR = DATA_DIR / "metadata"
RAG_DB_DIR = DATA_DIR / "rag_db"

# Ensure crucial directories exist
for folder in [RAW_PDFS_DIR, EXTRACTED_TEXT_DIR, CHUNKS_DIR, FINETUNING_DIR, METADATA_DIR, RAG_DB_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 2. Centralized Logging Config
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = get_logger("utils")

# 3. Topic Guessing/Classification Heuristics
TOPIC_KEYWORD_MAPPING = {
    "Retrieval-Augmented Generation (RAG)": [
        r"\brag\b", r"\bretrieval\b", r"\bvector\b", r"\bdatabase\b", r"\bchroma\b", 
        r"\bfaiss\b", r"\bsearch\b", r"\bembedding\b", r"\bdocuments?\b", r"\bpassage\b"
    ],
    "LLM Agents & Multi-Agent Systems": [
        r"\bagents?\b", r"\btools?\b", r"\breact\b", r"\bplanning\b", r"\bfunction\s+calling\b", 
        r"\bmulti-agent\b", r"\bactions?\b", r"\benvironment\b", r"\bdecision\b"
    ],
    "Fine-Tuning & PEFT (Parameter-Efficient Fine-Tuning)": [
        r"\blora\b", r"\bqlora\b", r"\bfine-tuning\b", r"\bsft\b", r"\binstruction\s+tuning\b", 
        r"\bpeft\b", r"\badapters?\b", r"\bparameter-efficient\b", r"\bsupervised\b"
    ],
    "LLM Alignment (RLHF / DPO)": [
        r"\brlhf\b", r"\bdpo\b", r"\balignment\b", r"\bppo\b", r"\bpreference\b", 
        r"\breward\s+models?\b", r"\bfeedback\b", r"\bhuman\s+in\s+the\s+loop\b"
    ],
    "Transformer Architecture & Attention": [
        r"\battention\b", r"\btransformers?\b", r"\bself-attention\b", r"\bmulti-head\b", 
        r"\bencoders?\b", r"\bdecoders?\b", r"\bfeedforward\b", r"\bkv\s+cache\b"
    ],
    "Tokenization & Vocabulary": [
        r"\btokens?\b", r"\btokenization\b", r"\bbpe\b", r"\bwordpiece\b", r"\bvocab\b", 
        r"\bbyte\s+pair\b", r"\btokenizer\b"
    ],
    "Prompt Engineering & In-Context Learning": [
        r"\bprompts?\b", r"\bfew-shot\b", r"\bzero-shot\b", r"\bcot\b", r"\bchain\s+of\s+thought\b", 
        r"\bin-context\b", r"\bsystem\s+prompts?\b", r"\binstruct\b"
    ],
    "LLM Pre-training & Training Objectives": [
        r"\bpre-train\b", r"\bpretraining\b", r"\bcausal\b", r"\bmasked\b", r"\bbert\b", 
        r"\bgpt\b", r"\bcorpus\b", r"\bpre-training\b"
    ],
    "LLM Evaluation & Safety": [
        r"\bevaluation\b", r"\bbenchmarks?\b", r"\bmmlu\b", r"\bgsm8k\b", r"\bmetrics?\b", 
        r"\bhallucinations?\b", r"\bbias\b", r"\bsafety\b", r"\btolerance\b"
    ]
}

def guess_topic(text: str) -> str:
    """
    Scans the text using regex keyword matches to assign a topic category.
    Returns the category with the highest number of keyword matches.
    If none match, returns 'Large Language Models Fundamentals'.
    """
    text_lower = text.lower()
    scores = {}
    
    for topic, patterns in TOPIC_KEYWORD_MAPPING.items():
        score = 0
        for pattern in patterns:
            # Find all occurrences
            matches = re.findall(pattern, text_lower)
            score += len(matches)
        if score > 0:
            scores[topic] = score
            
    if not scores:
        return "Large Language Models Fundamentals"
        
    # Get the topic with highest keyword matches
    best_topic = max(scores, key=scores.get)
    return best_topic
