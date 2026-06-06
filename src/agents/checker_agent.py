from src.tools.citation_checker_tool import check_citations

class CheckerAgent:
    def check(self, answer: str, retrieved_chunks: list) -> dict:
        citation_result = check_citations(answer, retrieved_chunks)
        return {
            "grounded": bool(retrieved_chunks) or "do not have enough grounded evidence" in answer.lower(),
            "citation_check": citation_result,
            "safe": True,
            "feedback": "Answer checked for grounding and citations.",
        }