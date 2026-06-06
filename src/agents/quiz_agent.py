from src.tools.quiz_tool import generate_quiz

class QuizAgent:
    def run(self, topic: str, difficulty: str = "medium", n: int = 3, context: str | None = None) -> dict:
        return generate_quiz(topic, difficulty, n, context)