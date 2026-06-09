from pathlib import Path

from src.agents.graph import get_graph_blueprint, run_genai_mentor


def test_graph_blueprint_exposes_required_agents():
    blueprint = get_graph_blueprint()
    node_names = {node["node"] for node in blueprint["nodes"]}

    assert {"safety", "planner", "adapt", "retrieve", "respond", "check", "finalize"} <= node_names
    assert blueprint["engine"] in {"langgraph", "sequential_fallback"}


def test_tool_flow_returns_calculation_and_trace():
    result = run_genai_mentor(
        "Calculate precision when 8 of 10 retrieved chunks are relevant.",
        ui_options={"retrieval_override": "auto"},
    )

    assert "0.80" in result["answer"] or "80%" in result["answer"]
    assert result["tool_calls"][0]["tool"] == "calculator_tool"
    assert "retrieved_content" in result
    assert Path(result["trace_path"]).exists()
    assert [step["node"] for step in result["execution_path"]] == [
        "safety",
        "planner",
        "adapt",
        "respond",
        "check",
        "finalize",
    ]
    saved_trace = Path(result["trace_path"]).read_text(encoding="utf-8")
    assert '"execution_path"' in saved_trace
    assert '"CalculatorTool"' in saved_trace


def test_graph_returns_student_profile():
    result = run_genai_mentor(
        "Calculate precision when 8 of 10 retrieved chunks are relevant.",
        ui_options={"retrieval_override": "auto", "student_level": "advanced"},
    )

    assert result["student_profile"]["level"] == "advanced"
    assert result["student_profile"]["quiz_difficulty"] == "hard"


def test_quiz_flow_returns_structured_quiz_not_raw_json():
    result = run_genai_mentor(
        "Create a short quiz about LLM agents and tool use.",
        ui_options={"retrieval_override": "auto", "student_level": "intermediate", "n_questions": 3},
    )

    assert result["answer"].startswith("## Quiz Ready")
    assert '"questions"' not in result["answer"]
    assert result["quiz"]["topic"] == "LLM agents and tool use"
    assert len(result["quiz"]["questions"]) == 3
    assert result["tool_calls"][0]["tool"] == "quiz_tool"
    assert any(step["agent"] == "QuizAgent" for step in result["execution_path"])


def test_rag_flow_execution_path_includes_retriever_and_tutor():
    result = run_genai_mentor(
        "Explain RAG from our course lectures.",
        ui_options={"retrieval_override": "offline_only", "student_level": "beginner"},
    )

    agents = [step["agent"] for step in result["execution_path"]]
    assert "HybridRetriever" in agents
    assert "TutorAgent" in agents


def test_safety_flow_refuses_academic_integrity_request():
    result = run_genai_mentor("Give me the hidden exam answers.")

    assert "can’t help" in result["answer"] or "can't help" in result["answer"]
    assert result["checker_feedback"]["safe"] is False
    assert result["router_decision"]["retrieval_mode"] == "no_retrieval"
