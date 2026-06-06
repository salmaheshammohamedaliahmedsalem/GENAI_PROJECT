import json


class LocalRuleBasedLLM:
    def generate(self, messages, temperature=0.2, tools=None) -> str:
        text = " ".join(m.get("content", "") for m in messages).lower()
        if "return json" in text and "retrieval_mode" in text:
            user_text = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                text,
            ).lower()
            return json.dumps(self._route(user_text))
        if "check if the answer" in text:
            return json.dumps({"grounded": True, "has_citations": "[" in text, "safe": True, "feedback": "Local checker passed."})
        if "grade" in text:
            return json.dumps({"score": 7, "max_score": 10, "strengths": ["Relevant attempt"], "mistakes": ["Needs more evidence"], "misconceptions": [], "feedback": "Good start. Add course citations.", "recommended_review_topics": ["RAG"]})
        if "quiz" in text:
            return json.dumps({"topic": "Generative AI", "difficulty": "medium", "questions": [{"type": "mcq", "question": "What does RAG add to an LLM?", "choices": ["Retrieval context", "More GPU only", "No prompts", "Randomness"], "answer": "Retrieval context", "explanation": "RAG grounds answers with retrieved information.", "source": "course chunks"}]})
        return "Based on the provided course sources, here is a grounded explanation. RAG retrieves relevant documents and adds them to the prompt so the model can answer with evidence. [Source: retrieved chunk]"

    def generate_json(self, messages, schema_hint=None, temperature=0.0) -> dict:
        raw = self.generate(messages, temperature=temperature)
        try:
            return json.loads(raw)
        except Exception:
            return {"text": raw}

    def _route(self, text: str) -> dict:
        unsafe = any(x in text for x in ["exam answers", "cheat", "hidden prompt", "ignore your rules"])
        if unsafe:
            return {"intent": "unsafe_or_policy_violation", "retrieval_mode": "no_retrieval", "needs_quiz": False, "needs_grading": False, "needs_tool": False, "needs_safety_check": True, "reasoning": "Unsafe request."}
        if any(x in text for x in ["calculate", "precision", "recall", "score"]):
            return {"intent": "calculation", "retrieval_mode": "tool_only", "needs_quiz": False, "needs_grading": False, "needs_tool": True, "needs_safety_check": True, "reasoning": "Calculation/tool request."}
        if "quiz" in text:
            return {"intent": "quiz_generation", "retrieval_mode": "tool_only", "needs_quiz": True, "needs_grading": False, "needs_tool": True, "needs_safety_check": True, "reasoning": "Quiz request."}
        if "grade" in text:
            return {"intent": "answer_grading", "retrieval_mode": "tool_only", "needs_quiz": False, "needs_grading": True, "needs_tool": True, "needs_safety_check": True, "reasoning": "Grading request."}
        if any(x in text for x in ["recent", "current", "latest", "modern improvement"]):
            mode = "hybrid" if any(x in text for x in ["lecture", "course", "our"]) else "online_only"
            return {"intent": "hybrid_course_plus_external", "retrieval_mode": mode, "needs_quiz": False, "needs_grading": False, "needs_tool": False, "needs_safety_check": True, "reasoning": "Needs current/external info."}
        return {"intent": "explain_course_concept", "retrieval_mode": "offline_only", "needs_quiz": False, "needs_grading": False, "needs_tool": False, "needs_safety_check": True, "reasoning": "Course concept."}
