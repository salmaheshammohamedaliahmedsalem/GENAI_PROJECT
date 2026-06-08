from src.llm.client import ChatClient
from src.llm.prompts import BASE_SYSTEM_PROMPT, TUTOR_PROMPT
from src.rag.citations import format_sources_for_prompt, citation_label

class TutorAgent:
    def __init__(self):
        self.llm = ChatClient()

    def answer(self, user_query: str, retrieved_chunks: list, retrieval_mode: str, student_profile: dict | None = None) -> str:
        student_profile = student_profile or {
            "level": "intermediate",
            "label": "Intermediate",
            "answer_style": "Use course terminology and explain mechanisms clearly.",
            "structure": "Give a concise explanation, key details, example, and next step.",
        }
        if retrieval_mode in {"offline_only", "hybrid"} and not retrieved_chunks:
            return "I do not have enough grounded evidence from the course sources to answer that confidently."

        sources = format_sources_for_prompt(retrieved_chunks)
        if self.llm.use_local:
            if retrieved_chunks:
                labels = " ".join(citation_label(r) for r in retrieved_chunks[:2])
                level = student_profile.get("label", "Intermediate")
                level_prefix = f"**Adapted for: {level}**\n\n"
                if retrieval_mode == "hybrid":
                    return level_prefix + (
                        "Course-grounded explanation:\n"
                        "RAG retrieves relevant course/source chunks and injects them into the prompt so the model answers using evidence. "
                        f"{labels}\n\nExternal update or project enhancement:\n"
                        "A modern improvement is to combine semantic search with keyword search and a checker agent.\n\n"
                        f"Learning focus: {student_profile.get('structure', '')}"
                    )
                if student_profile.get("level") == "beginner":
                    return level_prefix + (
                        f"RAG means Retrieval-Augmented Generation. In simple terms, the system first finds helpful course notes, then gives those notes to the model so the answer is based on evidence. {labels}\n\n"
                        "Analogy: it is like answering an exam with the correct textbook pages open beside you.\n\n"
                        "Quick check: why is using retrieved notes safer than guessing from memory?"
                    )
                if student_profile.get("level") == "advanced":
                    return level_prefix + (
                        f"RAG keeps model weights fixed and shifts factual grounding to retrieval-time context injection. The retriever ranks relevant chunks, the generator conditions on them, and the checker validates citation support. {labels}\n\n"
                        "Tradeoff: stronger grounding and updateability, but added retrieval latency and sensitivity to chunk quality.\n\n"
                        "Advanced follow-up: compare BM25-only retrieval with hybrid semantic + keyword retrieval."
                    )
                return level_prefix + (
                    f"RAG means Retrieval-Augmented Generation: retrieve relevant chunks, add them to the prompt, then generate a grounded answer. {labels}\n\n"
                    "Implementation view: retriever → prompt with evidence → generated answer → citation/checker step.\n\n"
                    "Next step: ask for a quiz on this topic."
                )
            return "I can explain this, but I do not have retrieved evidence available."

        return self.llm.generate([
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "system", "content": TUTOR_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {user_query}\n\n"
                    f"Retrieval mode: {retrieval_mode}\n\n"
                    f"Student profile: {student_profile}\n\n"
                    f"Sources:\n{sources}"
                ),
            },
        ])
