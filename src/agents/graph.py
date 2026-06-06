from datetime import datetime
import json
import re
from src.config import TRACE_DIR, ensure_dirs
from src.schemas import AgentTrace
from src.agents.safety_agent import check_safety
from src.agents.planner_agent import PlannerAgent
from src.agents.tutor_agent import TutorAgent
from src.agents.quiz_agent import QuizAgent
from src.agents.grader_agent import GraderAgent
from src.agents.checker_agent import CheckerAgent
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.citations import list_source_labels
from src.tools.calculator_tool import calculate_expression

def _extract_expression(text: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if m:
        return f"{m.group(1)} / {m.group(2)}"
    if "precision" in text.lower():
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if len(nums) >= 2:
            return f"{nums[0]} / {nums[1]}"
    return text

def _save_trace(trace: AgentTrace) -> str:
    ensure_dirs()
    path = TRACE_DIR / f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    return str(path)

def run_genai_mentor(user_query: str, conversation_history: list[dict] | None = None, ui_options: dict | None = None) -> dict:
    ui_options = ui_options or {}
    safety = check_safety(user_query)
    planner = PlannerAgent()
    retrieved = []
    tool_calls = []

    if not safety["safe"]:
        decision = planner.route(user_query)
        answer = safety["response"]
        checker_feedback = {"safe": False, "category": safety["category"]}
        trace = AgentTrace(user_query=user_query, router_decision=decision, retrieved_chunks=[], tool_calls=[], checker_feedback=checker_feedback, final_answer=answer)
        trace_path = _save_trace(trace)
        return {"answer": answer, "router_decision": decision.model_dump(), "sources": [], "tool_calls": [], "checker_feedback": checker_feedback, "trace_path": trace_path}

    override = ui_options.get("retrieval_override", "auto")
    decision = planner.route(user_query, override=override)

    if decision.retrieval_mode not in {"tool_only", "no_retrieval"}:
        retrieved = HybridRetriever().retrieve(user_query, mode=decision.retrieval_mode)

    if decision.needs_quiz or "quiz" in user_query.lower():
        context = "\n".join(r.chunk.text for r in retrieved)
        quiz = QuizAgent().run(user_query, ui_options.get("difficulty", "medium"), int(ui_options.get("n_questions", 3)), context)
        answer = "## Quiz\n\n" + json.dumps(quiz, indent=2, ensure_ascii=False)
        tool_calls.append({"tool": "quiz_tool", "args": {"topic": user_query}, "result": quiz})
    elif decision.needs_grading or "grade" in user_query.lower():
        result = GraderAgent().run("User-provided question", user_query)
        answer = "## Grading Feedback\n\n" + json.dumps(result, indent=2, ensure_ascii=False)
        tool_calls.append({"tool": "grading_tool", "result": result})
    elif decision.needs_tool or "calculate" in user_query.lower() or "precision" in user_query.lower():
        expr = _extract_expression(user_query)
        result = calculate_expression(expr)
        answer = f"The calculation is `{expr}`.\n\nResult: **{result.get('result')}**"
        if result.get("result") is not None and "precision" in user_query.lower():
            answer += f"\n\nPrecision = relevant retrieved / total retrieved = **{result.get('result'):.2f}** or **{result.get('result')*100:.0f}%**."
        tool_calls.append({"tool": "calculator_tool", "args": {"expression": expr}, "result": result})
    else:
        answer = TutorAgent().answer(user_query, retrieved, decision.retrieval_mode)

    checker_feedback = CheckerAgent().check(answer, retrieved)
    trace = AgentTrace(user_query=user_query, router_decision=decision, retrieved_chunks=retrieved, tool_calls=tool_calls, checker_feedback=checker_feedback, final_answer=answer)
    trace_path = _save_trace(trace)

    return {
        "answer": answer,
        "router_decision": decision.model_dump(),
        "sources": list_source_labels(retrieved),
        "tool_calls": tool_calls,
        "checker_feedback": checker_feedback,
        "trace_path": trace_path,
    }