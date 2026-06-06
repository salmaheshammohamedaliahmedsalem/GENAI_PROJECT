from src.agents.graph import run_genai_mentor

def run_baselines() -> list[dict]:
    questions = [
        "What is RAG and why does it reduce hallucinations?",
        "Explain LoRA from the course perspective.",
        "If 8 chunks are relevant out of 10 retrieved, calculate precision.",
    ]
    rows = []
    for q in questions:
        result = run_genai_mentor(q)
        rows.append({
            "question": q,
            "system": "final_system",
            "answer": result["answer"],
            "has_sources": bool(result.get("sources")),
            "has_tool_calls": bool(result.get("tool_calls")),
        })
    return rows