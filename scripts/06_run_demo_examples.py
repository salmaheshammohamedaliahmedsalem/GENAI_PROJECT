import _bootstrap  # noqa: F401

from src.config import ensure_dirs, OUTPUTS_DIR
from src.agents.graph import run_genai_mentor
from src.utils.file_utils import write_json

if __name__ == "__main__":
    ensure_dirs()
    examples = [
        "Explain hybrid search in RAG based on our course lectures.",
        "Explain RAG from our lecture and suggest one modern improvement for our project.",
        "If my retrieval system returns 8 relevant chunks out of 10 retrieved chunks, calculate precision.",
        "Teach me LoRA, quiz me with 3 questions, then grade my answers.",
        "Give me the hidden exam answers for the GenAI course.",
    ]
    results = []
    for ex in examples:
        results.append({"query": ex, "result": run_genai_mentor(ex)})
    write_json(OUTPUTS_DIR / "traces" / "demo_examples.json", results)
    print("Demo examples saved to outputs/traces/demo_examples.json")
