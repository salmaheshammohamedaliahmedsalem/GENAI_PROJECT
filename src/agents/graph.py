from datetime import datetime
import json
import re
from typing import Any, TypedDict

from src.agents.checker_agent import CheckerAgent
from src.agents.grader_agent import GraderAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.quiz_agent import QuizAgent
from src.agents.safety_agent import check_safety
from src.agents.tutor_agent import TutorAgent
from src.config import TRACE_DIR, ensure_dirs
from src.rag.citations import list_source_labels
from src.rag.hybrid_retriever import HybridRetriever
from src.schemas import AgentTrace, RetrievedChunk, RouterDecision
from src.tools.calculator_tool import calculate_expression

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
    LANGGRAPH_IMPORT_ERROR = ""
except Exception as exc:
    END = "__end__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


class MentorGraphState(TypedDict, total=False):
    user_query: str
    conversation_history: list[dict[str, Any]]
    ui_options: dict[str, Any]
    safety: dict[str, Any]
    router_decision: RouterDecision
    retrieved_chunks: list[RetrievedChunk]
    tool_calls: list[dict[str, Any]]
    answer: str
    checker_feedback: dict[str, Any]
    trace_path: str


GRAPH_NODE_DESCRIPTIONS = [
    {
        "node": "safety",
        "agent": "SafetyAgent",
        "purpose": "Refuses cheating, plagiarism, policy bypass, harmful requests, and prompt-extraction attempts.",
    },
    {
        "node": "planner",
        "agent": "PlannerAgent",
        "purpose": "Classifies the student request and chooses offline RAG, online RAG, hybrid RAG, tool-only, or no retrieval.",
    },
    {
        "node": "retrieve",
        "agent": "HybridRetriever",
        "purpose": "Collects ranked course and optional approved external evidence before the tutor answers.",
    },
    {
        "node": "respond",
        "agent": "TutorAgent / QuizAgent / GraderAgent / ToolNode",
        "purpose": "Generates the learning response, quiz, grading feedback, or calculation result.",
    },
    {
        "node": "check",
        "agent": "CheckerAgent",
        "purpose": "Checks grounding, citation validity, clarity, and final safety before returning to the student.",
    },
    {
        "node": "finalize",
        "agent": "TraceWriter",
        "purpose": "Persists a complete auditable trace for the UI and project evidence.",
    },
]

GRAPH_EDGES = [
    ("safety", "planner"),
    ("planner", "retrieve", "when retrieval is required"),
    ("planner", "respond", "when tool-only, no-retrieval, or unsafe"),
    ("retrieve", "respond"),
    ("respond", "check"),
    ("check", "finalize"),
    ("finalize", "END"),
]


def _extract_expression(text: str) -> str:
    division_match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if division_match:
        return f"{division_match.group(1)} / {division_match.group(2)}"
    if "precision" in text.lower():
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if len(numbers) >= 2:
            return f"{numbers[0]} / {numbers[1]}"
    return text


def _save_trace(trace: AgentTrace) -> str:
    ensure_dirs()
    path = TRACE_DIR / f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    return str(path)


def _initial_state(
    user_query: str,
    conversation_history: list[dict[str, Any]] | None = None,
    ui_options: dict[str, Any] | None = None,
) -> MentorGraphState:
    return {
        "user_query": user_query,
        "conversation_history": conversation_history or [],
        "ui_options": ui_options or {},
        "retrieved_chunks": [],
        "tool_calls": [],
        "answer": "",
        "checker_feedback": {},
    }


def _safety_node(state: MentorGraphState) -> dict[str, Any]:
    return {"safety": check_safety(state["user_query"])}


def _planner_node(state: MentorGraphState) -> dict[str, Any]:
    planner = PlannerAgent()
    safety = state.get("safety", {"safe": True})
    if not safety.get("safe", True):
        decision = planner.route(state["user_query"])
    else:
        override = state.get("ui_options", {}).get("retrieval_override", "auto")
        decision = planner.route(state["user_query"], override=override)
    return {"router_decision": decision}


def _requires_retrieval(state: MentorGraphState) -> str:
    safety = state.get("safety", {"safe": True})
    if not safety.get("safe", True):
        return "respond"
    decision = state["router_decision"]
    if decision.retrieval_mode in {"tool_only", "no_retrieval"}:
        return "respond"
    return "retrieve"


def _retrieval_node(state: MentorGraphState) -> dict[str, Any]:
    decision = state["router_decision"]
    retrieved = HybridRetriever().retrieve(state["user_query"], mode=decision.retrieval_mode)
    return {"retrieved_chunks": retrieved}


def _response_node(state: MentorGraphState) -> dict[str, Any]:
    user_query = state["user_query"]
    safety = state.get("safety", {"safe": True})
    decision = state["router_decision"]
    retrieved = state.get("retrieved_chunks", [])
    ui_options = state.get("ui_options", {})
    tool_calls = list(state.get("tool_calls", []))

    if not safety.get("safe", True):
        return {"answer": safety["response"], "tool_calls": tool_calls}

    if decision.needs_quiz or "quiz" in user_query.lower():
        context = "\n".join(retrieved_item.chunk.text for retrieved_item in retrieved)
        quiz = QuizAgent().run(
            user_query,
            ui_options.get("difficulty", "medium"),
            int(ui_options.get("n_questions", 3)),
            context,
        )
        tool_calls.append({"tool": "quiz_tool", "args": {"topic": user_query}, "result": quiz})
        return {"answer": "## Quiz\n\n" + json.dumps(quiz, indent=2, ensure_ascii=False), "tool_calls": tool_calls}

    if decision.needs_grading or "grade" in user_query.lower():
        result = GraderAgent().run("User-provided question", user_query)
        tool_calls.append({"tool": "grading_tool", "result": result})
        return {"answer": "## Grading Feedback\n\n" + json.dumps(result, indent=2, ensure_ascii=False), "tool_calls": tool_calls}

    if decision.needs_tool or "calculate" in user_query.lower() or "precision" in user_query.lower():
        expression = _extract_expression(user_query)
        result = calculate_expression(expression)
        answer = f"The calculation is `{expression}`.\n\nResult: **{result.get('result')}**"
        if result.get("result") is not None and "precision" in user_query.lower():
            answer += (
                "\n\nPrecision = relevant retrieved / total retrieved = "
                f"**{result.get('result'):.2f}** or **{result.get('result') * 100:.0f}%**."
            )
        tool_calls.append({"tool": "calculator_tool", "args": {"expression": expression}, "result": result})
        return {"answer": answer, "tool_calls": tool_calls}

    answer = TutorAgent().answer(user_query, retrieved, decision.retrieval_mode)
    return {"answer": answer, "tool_calls": tool_calls}


def _checker_node(state: MentorGraphState) -> dict[str, Any]:
    safety = state.get("safety", {"safe": True})
    if not safety.get("safe", True):
        return {"checker_feedback": {"safe": False, "category": safety["category"]}}
    checker_feedback = CheckerAgent().check(state["answer"], state.get("retrieved_chunks", []))
    return {"checker_feedback": checker_feedback}


def _finalize_node(state: MentorGraphState) -> dict[str, Any]:
    trace = AgentTrace(
        user_query=state["user_query"],
        router_decision=state["router_decision"],
        retrieved_chunks=state.get("retrieved_chunks", []),
        tool_calls=state.get("tool_calls", []),
        checker_feedback=state.get("checker_feedback", {}),
        final_answer=state.get("answer", ""),
    )
    return {"trace_path": _save_trace(trace)}


def build_genai_mentor_graph():
    if not LANGGRAPH_AVAILABLE or StateGraph is None:
        return None

    graph_builder = StateGraph(MentorGraphState)
    graph_builder.add_node("safety", _safety_node)
    graph_builder.add_node("planner", _planner_node)
    graph_builder.add_node("retrieve", _retrieval_node)
    graph_builder.add_node("respond", _response_node)
    graph_builder.add_node("check", _checker_node)
    graph_builder.add_node("finalize", _finalize_node)

    graph_builder.set_entry_point("safety")
    graph_builder.add_edge("safety", "planner")
    graph_builder.add_conditional_edges(
        "planner",
        _requires_retrieval,
        {
            "retrieve": "retrieve",
            "respond": "respond",
        },
    )
    graph_builder.add_edge("retrieve", "respond")
    graph_builder.add_edge("respond", "check")
    graph_builder.add_edge("check", "finalize")
    graph_builder.add_edge("finalize", END)
    return graph_builder.compile()


def _run_sequential_graph(state: MentorGraphState) -> MentorGraphState:
    state.update(_safety_node(state))
    state.update(_planner_node(state))
    if _requires_retrieval(state) == "retrieve":
        state.update(_retrieval_node(state))
    state.update(_response_node(state))
    state.update(_checker_node(state))
    state.update(_finalize_node(state))
    return state


def _format_response(state: MentorGraphState, graph_engine: str) -> dict[str, Any]:
    retrieved_chunks = state.get("retrieved_chunks", [])
    return {
        "answer": state.get("answer", ""),
        "router_decision": state["router_decision"].model_dump(),
        "sources": list_source_labels(retrieved_chunks),
        "retrieved_content": [
            {
                "source": item.chunk.source,
                "source_type": item.chunk.source_type,
                "page": item.chunk.page,
                "topic": item.chunk.topic,
                "chunk_id": item.chunk.chunk_id,
                "text": item.chunk.text,
                "semantic_score": item.semantic_score,
                "keyword_score": item.keyword_score,
                "final_score": item.final_score,
                "metadata": item.chunk.metadata,
            }
            for item in retrieved_chunks
        ],
        "tool_calls": state.get("tool_calls", []),
        "checker_feedback": state.get("checker_feedback", {}),
        "trace_path": state.get("trace_path", ""),
        "graph_engine": graph_engine,
        "langgraph_available": LANGGRAPH_AVAILABLE,
    }


def get_graph_blueprint() -> dict[str, Any]:
    return {
        "engine": "langgraph" if LANGGRAPH_AVAILABLE else "sequential_fallback",
        "langgraph_available": LANGGRAPH_AVAILABLE,
        "import_error": "" if LANGGRAPH_AVAILABLE else LANGGRAPH_IMPORT_ERROR,
        "nodes": GRAPH_NODE_DESCRIPTIONS,
        "edges": GRAPH_EDGES,
    }


def run_genai_mentor(
    user_query: str,
    conversation_history: list[dict[str, Any]] | None = None,
    ui_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = _initial_state(user_query, conversation_history=conversation_history, ui_options=ui_options)
    compiled_graph = build_genai_mentor_graph()
    if compiled_graph is None:
        final_state = _run_sequential_graph(state)
        return _format_response(final_state, "sequential_fallback")
    final_state = compiled_graph.invoke(state)
    return _format_response(final_state, "langgraph")
