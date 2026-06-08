from src.llm.client import ChatClient
from src.llm.model_registry import resolve_chat_model_option
from src.llm.prompts import BASE_SYSTEM_PROMPT, TUTOR_PROMPT
from src.rag.citations import format_sources_for_prompt, citation_label


def _ensure_source_labels(answer: str, retrieved_chunks: list) -> str:
    if not retrieved_chunks or "[Source:" in answer:
        return answer
    labels = " ".join(citation_label(chunk) for chunk in retrieved_chunks[:3])
    return f"{answer}\n\nSources used: {labels}"


class TutorAgent:
    def __init__(self):
        self.llm = ChatClient()

    def answer(
        self,
        user_query: str,
        retrieved_chunks: list,
        retrieval_mode: str,
        student_profile: dict | None = None,
        model_selection: str | None = None,
        conversation_history: list[dict] | None = None,
    ) -> str:
        student_profile = student_profile or {
            "level": "intermediate",
            "label": "Intermediate",
            "answer_style": "Use course terminology and explain mechanisms clearly.",
            "structure": "Give a concise explanation, key details, example, and next step.",
        }
        if retrieval_mode in {"offline_only", "hybrid"} and not retrieved_chunks:
            return "I do not have enough grounded evidence from the course sources to answer that confidently."

        sources = format_sources_for_prompt(retrieved_chunks)
        backend_kind = self.llm.backend_kind(model_selection)
        if backend_kind == "local_rule_based":
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

        selected_model = resolve_chat_model_option(model_selection)
        history_lines = []
        for message in (conversation_history or [])[-6:]:
            role = message.get("role", "user")
            content = message.get("content", "")
            if content:
                history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines) if history_lines else "No previous turns."

        messages = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "system", "content": TUTOR_PROMPT},
            {
                "role": "user",
                "content": (
                    "You are answering inside a RAG educational tutor. Use the retrieved context below as the factual basis.\n\n"
                    f"Selected response model: {selected_model.label}\n\n"
                    f"Student question: {user_query}\n\n"
                    f"Retrieval mode: {retrieval_mode}\n\n"
                    f"Student level/profile:\n{student_profile}\n\n"
                    f"Recent conversation:\n{history_text}\n\n"
                    "Retrieved RAG context sent to the model:\n"
                    f"{sources if sources else 'No retrieved RAG context was returned.'}\n\n"
                    "Required answer format:\n"
                    "- Start with a direct answer adapted to the student level.\n"
                    "- Use the retrieved context for factual claims.\n"
                    "- Include source citations exactly as provided after grounded claims.\n"
                    "- If the context is insufficient, say what evidence is missing."
                ),
            },
        ]
        try:
            answer = self.llm.generate(messages, model_selection=model_selection, max_new_tokens=512)
            return _ensure_source_labels(answer, retrieved_chunks)
        except Exception as exc:
            return (
                "The selected response model could not be loaded or used.\n\n"
                f"Model: `{selected_model.label}`\n\n"
                f"Reason: `{type(exc).__name__}: {exc}`\n\n"
                "Switch the response model to another available option, or install the local fine-tuning dependencies and ensure the base Qwen model is cached."
            )
