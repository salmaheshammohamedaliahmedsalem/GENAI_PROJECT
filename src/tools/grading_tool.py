def grade_answer(question: str, student_answer: str, reference_answer: str | None = None, rubric: dict | None = None) -> dict:
    answer = student_answer.lower()
    score = 0
    if len(answer.split()) >= 10:
        score += 3
    if any(x in answer for x in ["rag", "retrieval", "source", "citation", "ground"]):
        score += 4
    if reference_answer and any(word in answer for word in reference_answer.lower().split()[:10]):
        score += 2
    score = min(score, 10)
    return {
        "score": score,
        "max_score": 10,
        "strengths": ["Attempts to address the question"],
        "mistakes": [] if score >= 7 else ["Needs more precise course-grounded details"],
        "misconceptions": [],
        "feedback": "Good answer." if score >= 7 else "Revise using definitions, examples, and citations.",
        "recommended_review_topics": ["RAG", "Grounding", "Evaluation"],
    }