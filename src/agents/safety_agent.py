UNSAFE_TERMS = [
    "exam answers",
    "hidden exam",
    "cheat",
    "plagiarize",
    "write my final report so i can submit it as mine",
    "ignore your rules",
    "reveal your system prompt",
    "hack",
]

def check_safety(user_query: str) -> dict:
    q = user_query.lower()
    for term in UNSAFE_TERMS:
        if term in q:
            return {
                "safe": False,
                "category": "academic_integrity_or_policy",
                "response": "I can’t help with cheating, hidden exam answers, plagiarism, or bypassing course rules. I can help you study the topic, make a practice quiz, or explain the requirements safely.",
            }
    return {"safe": True, "category": "safe", "response": ""}