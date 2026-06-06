from src.tools.grading_tool import grade_answer

class GraderAgent:
    def run(self, question: str, student_answer: str, reference_answer: str | None = None, rubric: dict | None = None) -> dict:
        return grade_answer(question, student_answer, reference_answer, rubric)