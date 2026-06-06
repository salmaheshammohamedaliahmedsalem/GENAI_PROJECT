from src.config import OUTPUTS_DIR
from src.utils.file_utils import write_json

def compare_base_vs_tuned() -> None:
    rows = [
        {
            "prompt": "Explain RAG.",
            "base_model": "Generic answer without citations.",
            "tuned_model": "Course-style answer with citation behavior.",
            "ratings": {"course_alignment": 4, "clarity": 4, "citation_behavior": 4},
        }
    ]
    write_json(OUTPUTS_DIR / "finetune" / "base_vs_tuned_results.json", rows)