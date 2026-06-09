UNSAFE_TERMS = [
    # exam / answer leakage
    "exam answers",
    "hidden exam",
    # academic dishonesty — do/write work for me
    "cheat",
    "plagiarize",
    "write my essay",
    "write an essay",
    "write my assignment",
    "do my assignment",
    "write my report",
    "write a report for me",
    "write my paper",
    "do my homework",
    "write my homework",
    "complete my assignment",
    "finish my assignment",
    "write my final report so i can submit it as mine",
    # policy bypass
    "ignore your rules",
    "ignore your instructions",
    "reveal your system prompt",
    "forget your instructions",
    # harmful
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