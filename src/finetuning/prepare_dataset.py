import random
from src.config import PROCESSED_DIR, FINETUNE_DIR
from src.utils.jsonl_utils import read_jsonl, write_jsonl

TOPICS = ["RAG", "Embeddings", "Hybrid Search", "Prompt Design", "Transformers", "Fine-tuning", "PEFT/LoRA", "RLHF", "Agentic Workflows", "Tool Use", "Evaluation", "Ethics/Safety"]

def make_example(chunk: dict, topic: str) -> dict:
    source = chunk.get("source", "course source")
    page = chunk.get("page")
    citation = f"[Source: {source}" + (f", page {page}" if page else "") + f", chunk {chunk.get('chunk_id')}]"
    return {
        "instruction": f"Explain {topic} to a beginner using course style.",
        "context": chunk.get("text", "")[:1200],
        "response": f"{topic} is an important Generative AI concept. In this course context, it should be explained clearly, grounded in evidence, and connected to examples. {citation}",
        "topic": topic,
        "difficulty": "medium",
        "source": source,
    }

def prepare_finetune_dataset(max_examples: int | None = None) -> None:
    existing_sft = read_jsonl(FINETUNE_DIR / "sft_chat_dataset.jsonl")
    if existing_sft:
        examples = existing_sft[:max_examples] if max_examples else existing_sft
        random.seed(42)
        random.shuffle(examples)
        n = len(examples)
        write_jsonl(FINETUNE_DIR / "train.jsonl", examples[: int(0.8*n)])
        write_jsonl(FINETUNE_DIR / "val.jsonl", examples[int(0.8*n): int(0.9*n)])
        write_jsonl(FINETUNE_DIR / "test.jsonl", examples[int(0.9*n):])
        write_jsonl(FINETUNE_DIR / "sample_training_examples.jsonl", examples[:10])
        return

    chunks = read_jsonl(PROCESSED_DIR / "chunks.jsonl")
    if not chunks:
        chunks = [{"chunk_id": "seed_0", "text": "RAG retrieves relevant documents and uses them to ground LLM answers.", "source": "seed", "page": None}]
    examples = []
    fallback_count = max_examples or 300
    for i in range(fallback_count):
        chunk = chunks[i % len(chunks)]
        topic = TOPICS[i % len(TOPICS)]
        examples.append(make_example(chunk, topic))
    random.seed(42)
    random.shuffle(examples)
    n = len(examples)
    write_jsonl(FINETUNE_DIR / "train.jsonl", examples[: int(0.8*n)])
    write_jsonl(FINETUNE_DIR / "val.jsonl", examples[int(0.8*n): int(0.9*n)])
    write_jsonl(FINETUNE_DIR / "test.jsonl", examples[int(0.9*n):])
    write_jsonl(FINETUNE_DIR / "sample_training_examples.jsonl", examples[:10])
