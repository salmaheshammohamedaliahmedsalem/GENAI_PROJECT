from src.config import OUTPUTS_DIR
from src.utils.file_utils import read_json, write_json

PROGRESS_PATH = OUTPUTS_DIR / "progress.json"

def record_score(student_id: str, topic: str, score: float) -> dict:
    data = read_json(PROGRESS_PATH, default={}) or {}
    data.setdefault(student_id, []).append({"topic": topic, "score": score})
    write_json(PROGRESS_PATH, data)
    return {"ok": True, "student_id": student_id, "topic": topic, "score": score}

def get_weak_topics(student_id: str) -> list[str]:
    data = read_json(PROGRESS_PATH, default={}) or {}
    rows = data.get(student_id, [])
    return sorted({r["topic"] for r in rows if r.get("score", 10) < 6})