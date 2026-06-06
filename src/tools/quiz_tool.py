def generate_quiz(topic: str, difficulty: str = "medium", n: int = 3, context: str | None = None) -> dict:
    questions = []
    for i in range(n):
        questions.append({
            "type": "mcq",
            "question": f"{i+1}. What is an important idea about {topic}?",
            "choices": ["Grounding with sources", "Ignoring context", "Removing evaluation", "No safety"],
            "answer": "Grounding with sources",
            "explanation": f"{topic} should be explained using evidence and clear reasoning.",
            "source": "retrieved context" if context else "generated demo",
        })
    return {"topic": topic, "difficulty": difficulty, "questions": questions}