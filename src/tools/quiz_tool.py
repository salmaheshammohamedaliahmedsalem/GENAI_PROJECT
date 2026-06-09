def _clean_topic(topic: str) -> str:
    cleaned = topic.strip().rstrip(".?")
    prefixes = [
        "create a short quiz about ",
        "create a quiz about ",
        "make a short quiz about ",
        "make a quiz about ",
        "quiz me on ",
        "quiz me about ",
    ]
    lowered = cleaned.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    return cleaned.strip().rstrip(".?") or topic


def _question_bank(topic: str) -> list[dict]:
    return [
        {
            "type": "mcq",
            "question": f"Which statement best describes {topic}?",
            "choices": [
                "It should be grounded in relevant evidence and context.",
                "It should ignore retrieved information.",
                "It should remove evaluation and safety checks.",
                "It should rely only on memorized answers.",
            ],
            "answer": "It should be grounded in relevant evidence and context.",
            "explanation": f"{topic} should connect reasoning to evidence instead of producing unsupported claims.",
        },
        {
            "type": "mcq",
            "question": f"What is the safest way to use {topic} in an educational GenAI system?",
            "choices": [
                "Use tools with clear boundaries and inspectable outputs.",
                "Let the model take unrestricted actions.",
                "Hide sources from the student.",
                "Disable routing and safety checks.",
            ],
            "answer": "Use tools with clear boundaries and inspectable outputs.",
            "explanation": "Educational systems need controlled tools, visible evidence, and safety checks so students can trust and inspect the result.",
        },
        {
            "type": "mcq",
            "question": f"Why should {topic} be evaluated after generation?",
            "choices": [
                "To check correctness, grounding, and safety.",
                "To make answers longer automatically.",
                "To replace retrieval with guessing.",
                "To remove citations from the response.",
            ],
            "answer": "To check correctness, grounding, and safety.",
            "explanation": "Evaluation catches weak retrieval, unsupported claims, unsafe responses, and unclear explanations.",
        },
        {
            "type": "mcq",
            "question": f"What should a student look for when reviewing an answer about {topic}?",
            "choices": [
                "Clear reasoning, relevant sources, and explicit limitations.",
                "Only confident wording.",
                "No citations or trace details.",
                "Hidden tool outputs.",
            ],
            "answer": "Clear reasoning, relevant sources, and explicit limitations.",
            "explanation": "Good educational answers should be understandable, source-grounded, and honest about uncertainty.",
        },
    ]


def generate_quiz(topic: str, difficulty: str = "medium", n: int = 3, context: str | None = None) -> dict:
    clean_topic = _clean_topic(topic)
    source = "retrieved context" if context else "generated practice set"
    questions = []
    bank = _question_bank(clean_topic)
    for index in range(n):
        item = bank[index % len(bank)].copy()
        item["question"] = f"{index + 1}. {item['question']}"
        item["source"] = source
        questions.append(item)
    return {"topic": clean_topic, "difficulty": difficulty, "questions": questions}
