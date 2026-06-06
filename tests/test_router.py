from src.agents.planner_agent import PlannerAgent

def test_router_offline():
    r = PlannerAgent().route("Explain RAG from lecture")
    assert r.retrieval_mode in {"offline_only", "hybrid"}

def test_router_tool():
    r = PlannerAgent().route("calculate precision 8 out of 10")
    assert r.retrieval_mode == "tool_only"