from pathlib import Path

from src.agents.graph import get_graph_blueprint, run_genai_mentor


def test_graph_blueprint_exposes_required_agents():
    blueprint = get_graph_blueprint()
    node_names = {node["node"] for node in blueprint["nodes"]}

    assert {"safety", "planner", "retrieve", "respond", "check", "finalize"} <= node_names
    assert blueprint["engine"] in {"langgraph", "sequential_fallback"}


def test_tool_flow_returns_calculation_and_trace():
    result = run_genai_mentor(
        "Calculate precision when 8 of 10 retrieved chunks are relevant.",
        ui_options={"retrieval_override": "auto"},
    )

    assert "0.80" in result["answer"] or "80%" in result["answer"]
    assert result["tool_calls"][0]["tool"] == "calculator_tool"
    assert Path(result["trace_path"]).exists()


def test_safety_flow_refuses_academic_integrity_request():
    result = run_genai_mentor("Give me the hidden exam answers.")

    assert "can’t help" in result["answer"] or "can't help" in result["answer"]
    assert result["checker_feedback"]["safe"] is False
    assert result["router_decision"]["retrieval_mode"] == "no_retrieval"
