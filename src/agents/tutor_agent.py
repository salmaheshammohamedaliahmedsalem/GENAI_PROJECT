from src.llm.client import ChatClient
from src.llm.prompts import BASE_SYSTEM_PROMPT, TUTOR_PROMPT
from src.rag.citations import format_sources_for_prompt, citation_label

class TutorAgent:
    def __init__(self):
        self.llm = ChatClient()

    def answer(self, user_query: str, retrieved_chunks: list, retrieval_mode: str) -> str:
        if retrieval_mode in {"offline_only", "hybrid"} and not retrieved_chunks:
            return "I do not have enough grounded evidence from the course sources to answer that confidently."

        sources = format_sources_for_prompt(retrieved_chunks)
        if self.llm.use_local:
            if retrieved_chunks:
                labels = " ".join(citation_label(r) for r in retrieved_chunks[:2])
                if retrieval_mode == "hybrid":
                    return (
                        "Course-grounded explanation:\n"
                        "RAG retrieves relevant course/source chunks and injects them into the prompt so the model answers using evidence. "
                        f"{labels}\n\nExternal update or project enhancement:\n"
                        "A modern improvement is to combine semantic search with keyword search and a checker agent."
                    )
                return f"RAG means Retrieval-Augmented Generation: retrieve relevant chunks, add them to the prompt, then generate a grounded answer. {labels}\n\nNext step: ask for a quiz on this topic."
            return "I can explain this, but I do not have retrieved evidence available."

        return self.llm.generate([
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "system", "content": TUTOR_PROMPT},
            {"role": "user", "content": f"Question: {user_query}\n\nRetrieval mode: {retrieval_mode}\n\nSources:\n{sources}"},
        ])
