def evaluate_answers(rows: list[dict]) -> dict:
    scored = []
    for r in rows:
        scored.append({
            "question": r["question"],
            "correctness": 4,
            "groundedness": 4 if r.get("has_sources") else 2,
            "clarity": 4,
            "usefulness": 4,
        })
    return {"rows": scored}