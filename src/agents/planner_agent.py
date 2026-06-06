from src.llm.client import ChatClient
from src.llm.prompts import ROUTER_PROMPT
from src.schemas import RouterDecision

class PlannerAgent:
    def __init__(self):
        self.llm = ChatClient()

    def route(self, user_query: str, override: str = "auto") -> RouterDecision:
        if override != "auto":
            return RouterDecision(
                intent="manual_override",
                retrieval_mode=override,
                needs_quiz="quiz" in user_query.lower(),
                needs_grading="grade" in user_query.lower(),
                needs_tool=override == "tool_only",
                reasoning=f"Manual UI override: {override}",
            )
        data = self.llm.generate_json([
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": user_query},
        ])
        return RouterDecision(**{
            "intent": data.get("intent", "general_chat"),
            "retrieval_mode": data.get("retrieval_mode", "offline_only"),
            "needs_quiz": bool(data.get("needs_quiz", False)),
            "needs_grading": bool(data.get("needs_grading", False)),
            "needs_tool": bool(data.get("needs_tool", False)),
            "needs_safety_check": bool(data.get("needs_safety_check", True)),
            "reasoning": data.get("reasoning", ""),
        })