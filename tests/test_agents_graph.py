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


def test_safety_flow_refuses_academic_integrity_request():
    result = run_genai_mentor("Give me the hidden exam answers.")

    assert "can’t help" in result["answer"] or "can't help" in result["answer"]
    assert result["checker_feedback"]["safe"] is False
    assert result["router_decision"]["retrieval_mode"] == "no_retrieval"
