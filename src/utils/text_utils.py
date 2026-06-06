import re

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def infer_lecture_number(filename: str) -> int | None:
    match = re.search(r"lecture\s*(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else None

def infer_topic(filename: str, text: str = "") -> str | None:
    name = filename.lower()
    joined = f"{name} {text[:500].lower()}"
    topics = {
        "rag": "Retrieval Augmented Generation",
        "retrieval": "Retrieval Augmented Generation",
        "fine": "Fine-tuning",
        "lora": "PEFT / LoRA",
        "peft": "PEFT / LoRA",
        "agent": "Agentic AI",
        "transformer": "Transformer Architecture",
        "rlhf": "RLHF",
        "prompt": "Prompt Design",
        "scaling": "Pre-training and Scaling",
    }
    for key, topic in topics.items():
        if key in joined:
            return topic
    return None