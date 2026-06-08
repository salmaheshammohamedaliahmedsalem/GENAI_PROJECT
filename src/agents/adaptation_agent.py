LEVEL_PROFILES = {
    "beginner": {
        "label": "Beginner",
        "answer_style": "Use simple language, define terms before using them, avoid dense jargon, and include one analogy.",
        "structure": "Start with a short direct answer, then a simple example, then one quick check question.",
        "quiz_difficulty": "easy",
    },
    "intermediate": {
        "label": "Intermediate",
        "answer_style": "Use course terminology, explain mechanisms clearly, and connect ideas to implementation steps.",
        "structure": "Give a concise explanation, key technical details, a course example, and one next step.",
        "quiz_difficulty": "medium",
    },
    "advanced": {
        "label": "Advanced",
        "answer_style": "Use precise technical language, include tradeoffs, assumptions, and failure modes.",
        "structure": "Give a rigorous explanation, implementation implications, evaluation concerns, and an advanced follow-up.",
        "quiz_difficulty": "hard",
    },
}


class StudentAdaptationAgent:
    def run(self, student_level: str | None = None) -> dict:
        normalized = (student_level or "intermediate").strip().lower()
        if normalized not in LEVEL_PROFILES:
            normalized = "intermediate"
        profile = LEVEL_PROFILES[normalized].copy()
        profile["level"] = normalized
        return profile
