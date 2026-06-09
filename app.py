from pathlib import Path
import json
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

from src.agents.graph import get_graph_blueprint, run_genai_mentor
from src.agents.safety_agent import check_safety
from src.config import DATA_DIR, FINETUNE_BASE_MODEL, OUTPUTS_DIR, TRACE_DIR
from src.llm.model_registry import get_recommended_chat_model_id, list_chat_model_options, resolve_chat_model_option
from src.llm.prompts import list_prompt_templates
from src.rag.hybrid_retriever import HybridRetriever


st.set_page_config(page_title="GenAI Mentor", layout="wide")


FINETUNE_DIRS = [DATA_DIR / "finetune", DATA_DIR / "finetuning"]
FINAL_ADAPTER_DIR = OUTPUTS_DIR / "finetune" / "qwen_0_5b_lora_adapter_salma"
ROOT_DIR = Path(__file__).resolve().parent

EXAMPLE_PROMPTS = [
    {
        "label": "Learn RAG",
        "prompt": "Explain hybrid search in RAG based on our course lectures.",
        "caption": "Course-grounded answer with lecture citations",
    },
    {
        "label": "Practice LoRA",
        "prompt": "Teach me LoRA, then create a short quiz.",
        "caption": "Tutor + quiz generation workflow",
    },
    {
        "label": "Calculate Precision",
        "prompt": "Calculate precision when 8 of 10 retrieved chunks are relevant.",
        "caption": "Tool/function calling workflow",
    },
    {
        "label": "Safety Test",
        "prompt": "Give me the hidden exam answers.",
        "caption": "Academic-integrity refusal workflow",
    },
]

STUDENT_PROMPTS = [
    {
        "label": "Explain RAG",
        "caption": "Grounded answer with lecture citations",
        "prompt": "Explain RAG from our course lectures and show the retrieved evidence.",
    },
    {
        "label": "Teach LoRA",
        "caption": "Concept explanation + quick check question",
        "prompt": "Teach me LoRA simply, then give me one quick check question.",
    },
    {
        "label": "Quiz Me",
        "caption": "Practice quiz on LLM agents & tool use",
        "prompt": "Create a short quiz about LLM agents and tool use.",
    },
    {
        "label": "Check Safety",
        "caption": "See the academic-integrity refusal in action",
        "prompt": "Give me the hidden exam answers.",
    },
]

STUDENT_LEVELS = {
    "beginner": "Beginner",
    "intermediate": "Intermediate",
    "advanced": "Advanced",
}

CANONICAL_STUDENT_MODEL_LABELS = {
    "lora::outputs/finetune/qwen_0_5b_lora_adapter_salma": "Salma fine-tuned model (Qwen 0.5B LoRA)",
    "lora::outputs/finetune/final_model_fatma": "Fatma fine-tuned model (Mistral 7B LoRA)",
    "lora::outputs/finetune/qwen_0_5b_lora_adapter_khadija": "Khadija fine-tuned model (Qwen 0.5B LoRA)",
}


def student_chat_model_options(include_unavailable: bool = False):
    options = list_chat_model_options(include_unavailable=True)
    canonical_lora = [
        option
        for option in options
        if option.id in CANONICAL_STUDENT_MODEL_LABELS
        and (include_unavailable or option.available)
    ]
    non_lora = [
        option
        for option in options
        if option.kind != "lora_adapter"
        and (include_unavailable or option.available)
    ]
    return canonical_lora + non_lora


def student_chat_model_label(model_id: str) -> str:
    if model_id in CANONICAL_STUDENT_MODEL_LABELS:
        return CANONICAL_STUDENT_MODEL_LABELS[model_id]
    return resolve_chat_model_option(model_id).label


st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    .stApp {
        background: #f5f6f8;
        color: #111827;
    }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp li, .stApp td, .stApp th, .stApp caption {
        color: #111827;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div:hover {
        background-color: #ffffff !important;
        color: #111827 !important;
        border-color: #d1d5db !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #111827 !important;
        background-color: transparent;
    }
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #f3f4f6 !important;
    }
    div[data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    div[data-baseweb="menu"] li {
        color: #111827 !important;
    }
    div[data-baseweb="menu"] li:hover {
        background-color: #f3f4f6 !important;
    }
    .stSelectbox label, .stMultiSelect label, .stRadio label {
        color: #111827 !important;
    }
    div[data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border-color: #d1d5db !important;
    }
    div[data-testid="stFileUploader"] section > div,
    div[data-testid="stFileUploader"] section span,
    div[data-testid="stFileUploader"] section small,
    div[data-testid="stFileUploader"] section p,
    div[data-testid="stFileUploader"] label {
        color: #111827 !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;
        border-color: #d1d5db !important;
    }
    div[data-testid="stFileUploaderDropzone"] span,
    div[data-testid="stFileUploaderDropzone"] small {
        color: #111827 !important;
    }
    div[data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stFileUploader"] button {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
    }
    div[data-testid="stFileUploaderDropzone"] button:hover,
    div[data-testid="stFileUploader"] button:hover {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
        border-color: #9ca3af !important;
    }
    /* Apply border/background only to actual control buttons (secondary, primary, etc.) */
    button[data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-minimal"]) {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    button[data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-minimal"]):hover {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
        border-color: #9ca3af !important;
    }
    button[data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-minimal"]):active {
        background-color: #e5e7eb !important;
        color: #111827 !important;
    }
    button[data-testid^="stBaseButton-"] p,
    button[data-testid^="stBaseButton-"] span,
    button[data-testid^="stBaseButton-"] div {
        color: #111827 !important;
        background-color: transparent !important;
    }
    /* Minimal action buttons: transparent, no border */
    button[data-testid="stBaseButton-minimal"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 0.15rem !important;
        width: auto !important;
        min-width: unset !important;
    }
    button[data-testid="stBaseButton-minimal"]:hover {
        background: transparent !important;
        border: none !important;
    }
    button[data-testid="stBaseButton-minimal"] svg {
        stroke: #94a3b8 !important;
        fill: none !important;
        opacity: 1 !important;
    }
    .hero-card {
        padding: 1.1rem 1.4rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
        border: 1px solid #e2e8f0;
        margin-bottom: 0.65rem;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.07);
    }
    .hero-title {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.3rem;
    }
    .hero-title h1 {
        margin: 0;
        font-size: 1.7rem;
        font-weight: 700;
        color: #0f172a;
    }
    .small-muted {
        color: #64748b;
        font-size: 0.93rem;
        line-height: 1.5;
    }
    .boundary-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        margin-top: 0.55rem;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .step-card {
        min-height: 112px;
        padding: 0.9rem 1rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .step-card:hover {
        border-color: #c7d2fe;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }
    .step-card strong {
        display: block;
        margin-bottom: 0.3rem;
        color: #0f172a;
        font-size: 0.95rem;
    }
    .step-card-body {
        color: #64748b;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .success-pill {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: #ecfdf3;
        border: 1px solid #bbf7d0;
        color: #166534;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .student-card {
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid #dbeafe;
        background: #f8fbff;
        height: 100%;
    }
    .student-card h4 {
        margin: 0 0 0.35rem 0;
        color: #1d4ed8;
    }
    .agent-chip {
        display: inline-block;
        margin: 0.15rem 0.2rem 0.15rem 0;
        padding: 0.28rem 0.62rem;
        border-radius: 999px;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        color: #3730a3;
        font-weight: 600;
        font-size: 0.84rem;
    }
    .mode-toggle {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 0.65rem;
    }
    div[data-testid="stChatInput"] {
        border-radius: 18px;
    }
    div[data-testid="stChatInput"] > div {
        background-color: #ffffff !important;
        border: 1.5px solid #d1d5db !important;
        border-radius: 18px !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #111827 !important;
        caret-color: #111827 !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #9ca3af !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #eef0f4 !important;
        border-radius: 12px;
        padding: 0.2rem;
        gap: 0.15rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #4b5563 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.38rem 0.75rem !important;
        font-size: 0.9rem !important;
        transition: background-color 0.12s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #111827 !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.1) !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e2e5ea !important;
        color: #111827 !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    div[data-testid="stJson"] {
        background-color: #f8fafc !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
    }
    div[data-testid="stJson"] * {
        background-color: transparent !important;
    }
    div[data-testid="stJson"] span {
        color: #111827 !important;
    }
    .streamlit-json-container,
    .streamlit-json-container * {
        background-color: #f8fafc !important;
        color: #111827 !important;
    }
    pre[class*="language-"],
    code[class*="language-"] {
        background-color: #f8fafc !important;
        color: #111827 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def latest_file(path: Path, pattern: str) -> Path | None:
    files = sorted(path.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True) if path.exists() else []
    return files[0] if files else None


def finetune_counts() -> list[dict]:
    rows = []
    for directory in FINETUNE_DIRS:
        if not directory.exists():
            continue
        for name in [
            "tutor_dataset_clean.jsonl",
            "examiner_dataset_clean.jsonl",
            "critic_dataset_clean.jsonl",
            "combined_dataset_clean.jsonl",
            "sft_chat_dataset.jsonl",
            "train.jsonl",
            "val.jsonl",
            "test.jsonl",
        ]:
            path = directory / name
            if path.exists():
                rows.append({"directory": str(directory), "dataset": name, "examples": count_jsonl(path)})
    return rows


def component_status() -> list[dict]:
    return [
        {"component": "Prompt Design", "evidence": "src/llm/prompts.py", "status": "Implemented"},
        {"component": "Offline/Hybrid RAG", "evidence": "src/rag/ + data/processed/bm25_index.pkl", "status": "Implemented with BM25; semantic Chroma is optional"},
        {"component": "Fine-tuning/PEFT", "evidence": "src/finetuning/ + outputs/finetune/qwen_0_5b_lora_adapter_salma", "status": "Qwen LoRA adapter trained on MPS"},
        {"component": "Tools/Function Calling", "evidence": "src/tools/", "status": "Implemented"},
        {
            "component": "LangGraph Multi-Agent System",
            "evidence": "src/agents/graph.py + requirements.txt",
            "status": "Implemented with real LangGraph when installed and the same local sequential graph fallback otherwise",
        },
        {"component": "Evaluation", "evidence": "src/evaluation/ + outputs/evaluation", "status": "Implemented, run report before submission"},
        {"component": "Safety/Ethics", "evidence": "src/agents/safety_agent.py + docs/ethics_safety.md", "status": "Implemented"},
        {"component": "GUI Demo", "evidence": "app.py", "status": "Working showcase"},
    ]


def run_command(command: list[str], env: dict[str, str] | None = None, timeout: int = 120) -> dict:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(command),
            "returncode": -1,
            "stdout": exc.stdout or "",
            "stderr": f"Timed out after {timeout} seconds.\n{exc.stderr or ''}",
        }


def show_command_result(result: dict) -> None:
    status = "passed" if result["returncode"] == 0 else "failed"
    if result["returncode"] == 0:
        st.success(f"{status.title()} with return code {result['returncode']}")
    else:
        st.error(f"{status.title()} with return code {result['returncode']}")
    st.caption(f"Command: `{result['command']}`")
    if result["stdout"]:
        st.code(result["stdout"], language="text")
    if result["stderr"]:
        st.code(result["stderr"], language="text")


def dot_escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def fallback_execution_path_from_trace(trace: dict) -> list[dict]:
    if trace.get("execution_path"):
        return trace["execution_path"]
    decision = trace.get("router_decision", {})
    tool_calls = trace.get("tool_calls", [])
    retrieved = trace.get("retrieved_chunks", [])
    response_agent = "TutorAgent"
    tool_names = [call.get("tool", "") for call in tool_calls]
    if any(name == "quiz_tool" for name in tool_names):
        response_agent = "QuizAgent"
    elif any(name == "grading_tool" for name in tool_names):
        response_agent = "GraderAgent"
    elif any(name == "calculator_tool" for name in tool_names):
        response_agent = "CalculatorTool"
    elif decision.get("retrieval_mode") == "no_retrieval":
        response_agent = "SafetyAgent"

    steps = [
        {"order": 1, "node": "safety", "agent": "SafetyAgent", "status": "completed", "detail": "request checked"},
        {
            "order": 2,
            "node": "planner",
            "agent": "PlannerAgent",
            "status": decision.get("retrieval_mode", "unknown"),
            "detail": f"intent={decision.get('intent', 'unknown')}",
        },
        {"order": 3, "node": "adapt", "agent": "StudentAdaptationAgent", "status": "completed", "detail": "student profile selected"},
    ]
    if retrieved and decision.get("retrieval_mode") not in {"tool_only", "no_retrieval"}:
        steps.append(
            {
                "order": len(steps) + 1,
                "node": "retrieve",
                "agent": "HybridRetriever",
                "status": f"{len(retrieved)} chunks",
                "detail": "retrieved evidence before response",
            }
        )
    for node, agent, status, detail in [
        ("respond", response_agent, "generated", "response produced"),
        ("check", "CheckerAgent", "completed", "final answer checked"),
        ("finalize", "TraceWriter", "saved", "trace saved"),
    ]:
        steps.append({"order": len(steps) + 1, "node": node, "agent": agent, "status": status, "detail": detail})
    return steps


def latest_trace_payload() -> tuple[Path | None, dict]:
    latest = latest_file(TRACE_DIR, "trace_*.json")
    if latest is None:
        return None, {}
    try:
        return latest, json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return latest, {}


def latest_execution_payload() -> tuple[str, dict]:
    live_result = st.session_state.get("last_agent_result")
    if live_result:
        return "Latest live app response", live_result
    latest, trace = latest_trace_payload()
    if trace:
        return str(latest), trace
    return "", {}


def agent_graph_dot(execution_path: list[dict] | None = None) -> str:
    blueprint = get_graph_blueprint()
    active_nodes = {step.get("node") for step in execution_path or []}
    active_agents = {step.get("node"): step.get("agent") for step in execution_path or []}
    path_nodes = [step.get("node") for step in (execution_path or []) if step.get("node")]
    active_edges = set(zip(path_nodes, path_nodes[1:]))
    lines = [
        "digraph GenAIMentor {",
        "  rankdir=LR;",
        '  graph [bgcolor="transparent", pad="0.2", nodesep="0.45", ranksep="0.65"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.12,0.08"];',
        '  edge [fontname="Helvetica", fontsize=9, color="#94a3b8", arrowsize=0.8];',
    ]
    for node in blueprint["nodes"]:
        node_id = node["node"]
        active = node_id in active_nodes
        agent = active_agents.get(node_id) or node["agent"]
        label = f"{node_id}\\n{agent}"
        fill = "#fff7ed" if active else "#f8fafc"
        color = "#f97316" if active else "#cbd5e1"
        penwidth = "2.6" if active else "1.2"
        lines.append(
            f'  "{dot_escape(node_id)}" [label="{dot_escape(label)}", fillcolor="{fill}", color="{color}", penwidth={penwidth}];'
        )
    for edge in blueprint["edges"]:
        source, target = edge[0], edge[1]
        condition = edge[2] if len(edge) > 2 else ""
        active = (source, target) in active_edges
        color = "#f97316" if active else "#94a3b8"
        penwidth = "2.8" if active else "1.2"
        label = f' [label="{dot_escape(condition)}", color="{color}", fontcolor="{color}", penwidth={penwidth}]' if condition else f' [color="{color}", penwidth={penwidth}]'
        lines.append(f'  "{dot_escape(source)}" -> "{dot_escape(target)}"{label};')
    lines.append("}")
    return "\n".join(lines)


def render_agent_graph(execution_path: list[dict] | None = None) -> None:
    dot = agent_graph_dot(execution_path)
    try:
        st.graphviz_chart(dot)
    except Exception:
        st.code(dot, language="dot")


def render_execution_path(execution_path: list[dict]) -> None:
    if not execution_path:
        st.info("No execution path is available yet. Run a student chat or backend demo request first.")
        return
    st.markdown(
        " → ".join(
            f"<span class='agent-chip'>{step.get('order', index)}. {step.get('agent', step.get('node', 'agent'))}</span>"
            for index, step in enumerate(execution_path, start=1)
        ),
        unsafe_allow_html=True,
    )
    for index, step in enumerate(execution_path, start=1):
        with st.container(border=True):
            st.markdown(f"**Step {step.get('order', index)} — {step.get('agent', 'Agent')}**")
            st.caption(f"Graph node: `{step.get('node', '')}` · Status: `{step.get('status', '')}`")
            st.write(step.get("detail", ""))
    rows = [
        {
            "step": step.get("order", index),
            "node": step.get("node", ""),
            "agent/tool": step.get("agent", ""),
            "status": step.get("status", ""),
            "what happened": step.get("detail", ""),
        }
        for index, step in enumerate(execution_path, start=1)
    ]
    with st.expander("Compact execution table"):
        st.table(rows)


def result_caption(result: dict, fallback_model_label: str | None = None) -> str:
    decision = result.get("router_decision", {})
    response_model = result.get("response_model", {})
    profile = result.get("student_profile", {})
    parts = [
        f"Route: `{decision.get('retrieval_mode', 'unknown')}`",
        f"Level: `{profile.get('label', 'Adaptive')}`",
    ]
    model_label = response_model.get("label") or fallback_model_label
    if model_label:
        parts.append(f"Model: `{model_label}`")
    parts.append(f"Graph: `{result.get('graph_engine', 'unknown')}`")
    return " · ".join(parts)


def render_quiz_card(quiz: dict) -> None:
    questions = quiz.get("questions", [])
    st.markdown("### 🧠 Practice Quiz")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Topic", str(quiz.get("topic", "Practice"))[:28])
    metric_cols[1].metric("Difficulty", str(quiz.get("difficulty", "medium")).title())
    metric_cols[2].metric("Questions", len(questions))
    st.caption("Try answering each question first, then open the answer panel to check yourself.")

    for index, question in enumerate(questions, start=1):
        with st.container(border=True):
            question_text = question.get("question", f"Question {index}")
            st.markdown(f"#### Question {index}")
            st.markdown(question_text)
            choices = question.get("choices", [])
            for choice_index, choice in enumerate(choices):
                letter = chr(ord("A") + choice_index)
                st.markdown(f"- **{letter}.** {choice}")
            with st.expander("Show answer and explanation"):
                st.success(f"Answer: {question.get('answer', 'Not provided')}")
                explanation = question.get("explanation")
                if explanation:
                    st.write(explanation)
                source = question.get("source")
                if source:
                    st.caption(f"Source basis: {source}")


def legacy_quiz_from_content(content: str) -> dict:
    stripped = content.strip()
    if not stripped.startswith("## Quiz"):
        return {}
    json_start = stripped.find("{")
    if json_start < 0:
        return {}
    try:
        quiz = json.loads(stripped[json_start:])
    except Exception:
        return {}
    return quiz if isinstance(quiz, dict) else {}


def render_assistant_content(content: str) -> None:
    quiz = legacy_quiz_from_content(content)
    if quiz.get("questions"):
        st.markdown("## Quiz Ready")
        render_quiz_card(quiz)
    else:
        st.markdown(content)


def render_assistant_result(result: dict, fallback_model_label: str | None = None) -> None:
    quiz = result.get("quiz") or {}
    if quiz.get("questions"):
        st.markdown(result.get("answer", "I created a practice quiz."))
        render_quiz_card(quiz)
    else:
        st.markdown(result.get("answer", ""))
    st.caption(result_caption(result, fallback_model_label=fallback_model_label))


def show_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-title">
            <span style="font-size:1.9rem;line-height:1;">🎓</span>
            <h1>GenAI Mentor</h1>
          </div>
          <div class="small-muted">
            Student-first Generative AI learning system — grounded explanations, practice quizzes,
            grading feedback, citations, agent traces, LoRA evidence, and academic-integrity guardrails.
          </div>
          <span class="boundary-pill">📚 Helps students learn · never leaks exam answers · never fabricates citations</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_learning_steps() -> None:
    steps = [
        ("1. Ask", "A student asks a course question, requests practice, or submits an answer."),
        ("2. Route", "The planner chooses offline RAG, online RAG, hybrid RAG, a tool, quiz, or safety flow."),
        ("3. Teach", "Tutor, quiz, grader, and checker agents produce a grounded learning response."),
        ("4. Review", "The GUI shows sources, trace, evaluation, safety checks, and fine-tuning evidence."),
    ]
    cols = st.columns(4)
    for col, (title, body) in zip(cols, steps):
        with col:
            st.markdown(f"<div class='step-card'><strong>{title}</strong><span class='step-card-body'>{body}</span></div>", unsafe_allow_html=True)


def show_mode_selector() -> None:
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "student"

    student_col, backend_col = st.columns(2)
    with student_col:
        if st.button(
            "🎓 Student",
            type="primary" if st.session_state.view_mode == "student" else "secondary",
            width="stretch",
        ):
            st.session_state.view_mode = "student"
            st.rerun()
    with backend_col:
        if st.button(
            "🛠️ Backend Tracking",
            type="primary" if st.session_state.view_mode == "backend" else "secondary",
            width="stretch",
        ):
            st.session_state.view_mode = "backend"
            st.rerun()


def show_retrieved_content_panel(result: dict | None) -> None:
    st.markdown("### Retrieved sources")
    st.caption("What the assistant used for the latest answer.")

    if not result:
        st.markdown(
            """
            <div style="text-align:center; padding: 2rem 0.5rem; color: #6b7280;">
              <div style="font-size:2.2rem; margin-bottom:0.6rem;">🔍</div>
              <div style="font-weight:600; color:#374151; margin-bottom:0.3rem;">No sources yet</div>
              <div style="font-size:0.88rem;">Ask a question and the evidence chunks used to answer it will appear here — with source, page, topic, and relevance score.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    decision = result.get("router_decision", {})
    route = decision.get("retrieval_mode", "unknown")
    intent = decision.get("intent", "unknown")
    profile = result.get("student_profile", {})
    route_col, intent_col, level_col = st.columns(3)
    route_col.metric("Route", route)
    intent_col.metric("Intent", intent[:22] + ("..." if len(intent) > 22 else ""))
    level_col.metric("Level", profile.get("label", "Adaptive"))

    retrieved = result.get("retrieved_content", [])
    if not retrieved:
        if result.get("quiz"):
            quiz = result["quiz"]
            st.info("This request used the QuizAgent instead of retrieval.")
            quiz_cols = st.columns(2)
            quiz_cols[0].metric("Difficulty", str(quiz.get("difficulty", "medium")).title())
            quiz_cols[1].metric("Questions", len(quiz.get("questions", [])))
            with st.expander("Quiz data used by backend"):
                st.json(quiz)
            return
        if result.get("tool_calls"):
            st.info("This request used a tool instead of retrieval.")
            st.json(result.get("tool_calls"))
        else:
            st.info("No retrieved content was needed or found for the latest answer.")
        return

    for index, item in enumerate(retrieved, start=1):
        title = item.get("metadata", {}).get("title") or item.get("topic") or f"Source {index}"
        score = item.get("final_score")
        score_text = f" · score {score:.2f}" if isinstance(score, (int, float)) else ""
        page = f" · page {item.get('page')}" if item.get("page") else ""
        topic = f" · {item.get('topic')}" if item.get("topic") else ""
        source = item.get("source") or "unknown source"
        preview = (item.get("text", "") or "").strip()
        preview = preview[:300] + ("..." if len(preview) > 300 else "")
        with st.container(border=True):
            st.markdown(f"**{index}. {title}**")
            st.caption(f"{item.get('source_type')} · {source}{page}{topic}{score_text}")
            st.write(preview)
            with st.expander("Full retrieved text"):
                st.write(item.get("text", ""))


def _render_pdf_upload_panel() -> str | None:
    """Renders the PDF upload widget. Returns the session_collection name if a PDF is loaded, else None."""
    st.divider()
    st.markdown("**Upload a study PDF**")
    st.caption("Your PDF will be indexed and searched alongside course material.")

    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader", label_visibility="collapsed")

    if uploaded is not None:
        collection_name = "session_pdf"
        prev_name = st.session_state.get("pdf_collection_name")
        prev_file = st.session_state.get("pdf_uploaded_name")

        if prev_file != uploaded.name or prev_name != collection_name:
            with st.spinner("Indexing PDF..."):
                try:
                    from src.rag.pdf_ingestor import ingest_pdf
                    result = ingest_pdf(uploaded.read(), collection_name=collection_name)
                except ImportError:
                    result = {"status": "error", "error": "pdf_ingestor not available", "chunks": 0}
            st.session_state.pdf_ingest_result = result
            st.session_state.pdf_collection_name = collection_name
            st.session_state.pdf_uploaded_name = uploaded.name

        result = st.session_state.get("pdf_ingest_result", {})
        if result.get("status") == "ok":
            st.success(f"Ready — {result['chunks']} chunks indexed from **{uploaded.name}**")
        elif result.get("status") == "error":
            st.error(f"Failed: {result.get('error', 'unknown error')}")
        elif result.get("status") == "empty":
            st.warning("No text could be extracted from this PDF.")

        if st.button("Clear PDF", key="clear_pdf", width="stretch"):
            try:
                from src.rag.pdf_ingestor import clear_pdf_collection
                clear_pdf_collection(collection_name)
            except Exception:
                pass
            for key in ("pdf_ingest_result", "pdf_collection_name", "pdf_uploaded_name"):
                st.session_state.pop(key, None)
            st.rerun()

        if result.get("status") == "ok":
            return st.session_state.get("pdf_collection_name")

    elif st.session_state.get("pdf_collection_name"):
        for key in ("pdf_ingest_result", "pdf_collection_name", "pdf_uploaded_name"):
            st.session_state.pop(key, None)

    return None


def show_student_view() -> None:
    if "student_messages" not in st.session_state:
        st.session_state.student_messages = []

    st.markdown("## Student workspace")
    st.caption("Ask questions, get course-grounded answers, and inspect the sources used.")

    rail_col, chat_col, sources_col = st.columns([0.23, 0.52, 0.25], gap="large")

    with rail_col:
        with st.container(border=True):
            st.markdown("### Study tools")
            st.caption("Use a starter prompt or begin a fresh chat.")
            selected_level = st.selectbox(
                "Your level",
                options=list(STUDENT_LEVELS.keys()),
                index=1,
                format_func=lambda value: STUDENT_LEVELS[value],
            )
            model_options = student_chat_model_options()
            model_ids = [option.id for option in model_options]
            recommended_model_id = get_recommended_chat_model_id()
            model_index = model_ids.index(recommended_model_id) if recommended_model_id in model_ids else 0
            selected_model_id = st.selectbox(
                "Response model",
                options=model_ids,
                index=model_index,
                format_func=student_chat_model_label,
            )
            selected_model = resolve_chat_model_option(selected_model_id)
            st.caption(selected_model.status)
            if selected_model.is_finetuned:
                if selected_model.available:
                    st.success("Using a runnable fine-tuned LoRA tutor model.")
                else:
                    st.warning("Saved adapter is present, but this environment cannot run it until `requirements_finetune.txt` dependencies and the required base model cache/download setting are available.")
            if st.button("New chat", key="clear_student_chat", width="stretch"):
                st.session_state.student_messages = []
                st.session_state.last_student_result = None
                st.rerun()
            st.divider()
            st.caption("Starter prompts")
            for index, item in enumerate(STUDENT_PROMPTS):
                with st.container(border=True):
                    if st.button(
                        item["label"],
                        key=f"student_prompt_{index}",
                        width="stretch",
                    ):
                        st.session_state.pending_student_query = item["prompt"]
                    st.caption(item["caption"])
            session_collection = _render_pdf_upload_panel()
            st.divider()
            st.caption("Sources stay visible on the right. Backend details are separated into Backend Tracking.")

    with chat_col:
        with st.container(border=True):
            header_col, status_col = st.columns([0.72, 0.28])
            header_col.markdown("### Chat")
            status_col.success("Ready")

            pdf_name = st.session_state.get("pdf_uploaded_name")
            pdf_status = st.session_state.get("pdf_ingest_result", {}).get("status")
            if pdf_name and pdf_status == "ok":
                short_name = pdf_name if len(pdf_name) <= 28 else pdf_name[:25] + "..."
                st.markdown(
                    f"<div style='display:inline-flex;align-items:center;gap:0.4rem;"
                    f"background:#ecfdf5;border:1px solid #6ee7b7;border-radius:999px;"
                    f"padding:0.18rem 0.65rem;font-size:0.82rem;color:#065f46;margin-bottom:0.5rem;'>"
                    f"📄 <strong>{short_name}</strong> active</div>",
                    unsafe_allow_html=True,
                )

            if not st.session_state.student_messages:
                st.info('Start with a question like: "Explain RAG using course sources."')

            for message in st.session_state.student_messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant" and message.get("result"):
                        render_assistant_result(message["result"])
                    elif message["role"] == "assistant":
                        render_assistant_content(message["content"])
                    else:
                        st.markdown(message["content"])

            query = st.session_state.pop("pending_student_query", None)
            typed_query = st.chat_input("Ask a GenAI course question, request a quiz, or submit an answer for feedback.")
            query = query or typed_query

            if query:
                st.session_state.student_messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)

                with st.chat_message("assistant"):
                    with st.spinner("Retrieving evidence and preparing a learning response..."):
                        result = run_genai_mentor(
                            query,
                            conversation_history=st.session_state.student_messages,
                            ui_options={
                                "retrieval_override": "auto",
                                "student_level": selected_level,
                                "chat_model_id": selected_model_id,
                                "n_questions": 3,
                                "session_collection": session_collection,
                            },
                        )
                    render_assistant_result(result, fallback_model_label=selected_model.label)
                st.session_state.student_messages.append({"role": "assistant", "content": result["answer"], "result": result})
                st.session_state.last_student_result = result
                st.session_state.last_agent_result = result

    with sources_col:
        with st.container(border=True):
            show_retrieved_content_panel(st.session_state.get("last_student_result"))


def show_chat_tab() -> None:
    with st.sidebar:
        st.header("Teaching Controls")
        st.caption("Use these controls to demonstrate how the educational agents behave.")
        retrieval_override = st.selectbox(
            "Retrieval mode",
            ["auto", "offline_only", "hybrid", "online_only", "tool_only"],
        )
        difficulty = st.selectbox("Quiz difficulty", ["easy", "medium", "hard"], index=1)
        n_questions = st.number_input("Quiz questions", min_value=1, max_value=10, value=3)
        show_trace = st.checkbox("Show agent trace", value=True)
        if st.button("Clear chat", width="stretch"):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.markdown("**Suggested demo path**")
        st.markdown("1. Explain a concept\n2. Generate quiz\n3. Grade/check\n4. Show safety refusal")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.subheader("Guided Learning Chat")
    st.caption("Pick a classroom scenario or type your own question. The system will route through the agents.")
    example_cols = st.columns(len(EXAMPLE_PROMPTS))
    for index, item in enumerate(EXAMPLE_PROMPTS):
        with example_cols[index]:
            if st.button(item["label"], key=f"example_prompt_{index}", width="stretch"):
                st.session_state.pending_query = item["prompt"]
            st.caption(item["caption"])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("result"):
                render_assistant_result(message["result"])
            elif message["role"] == "assistant":
                render_assistant_content(message["content"])
            else:
                st.markdown(message["content"])

    query = st.session_state.pop("pending_query", None)
    typed_query = st.chat_input("Ask about GenAI, RAG, LoRA, agents, project requirements, or ask for a quiz.")
    query = query or typed_query

    if not query:
        st.info('Start with **Learn RAG** or ask: "Teach me LoRA and quiz me."')
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Agents are working..."):
            result = run_genai_mentor(
                query,
                conversation_history=st.session_state.messages,
                ui_options={
                    "retrieval_override": retrieval_override,
                    "difficulty": difficulty,
                    "n_questions": int(n_questions),
                },
            )
        render_assistant_result(result)
        decision = result.get("router_decision", {})
        intent = decision.get("intent", "unknown")
        st.caption(f"Intent: `{intent}`")

        if result.get("sources"):
            with st.expander("Evidence sources used"):
                for source in result["sources"]:
                    st.write(source)

        if show_trace:
            with st.expander("Agent trace and checker feedback"):
                st.json({
                    "graph_engine": result.get("graph_engine"),
                    "router_decision": result.get("router_decision"),
                    "tool_calls": result.get("tool_calls"),
                    "checker_feedback": result.get("checker_feedback"),
                    "trace_path": result.get("trace_path"),
                })

    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "result": result})
    st.session_state.last_backend_result = result
    st.session_state.last_agent_result = result


def show_overview_tab() -> None:
    st.subheader("Educational System Overview")
    st.write(
        "This is not just a PDF chatbot. It is a learning loop: retrieve course evidence, teach clearly, "
        "quiz the student, grade answers, check grounding, and refuse unsafe academic-integrity requests."
    )
    blueprint = get_graph_blueprint()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Course PDFs", len(list((DATA_DIR / "raw/course_pdfs").glob("*.pdf"))))
    col2.metric("Lecture Chunks", count_jsonl(DATA_DIR / "chunks/lecture_chunks.jsonl"))
    col3.metric("SFT Examples", count_jsonl(DATA_DIR / "finetune/sft_chat_dataset.jsonl"))
    col4.metric("Trace Files", len(list(TRACE_DIR.glob("*.json"))) if TRACE_DIR.exists() else 0)
    col5.metric("Graph Engine", blueprint["engine"])

    st.markdown("### Student Learning Flow")
    show_learning_steps()

    st.markdown("### What Students Can Do")
    student_cols = st.columns(3)
    student_cards = [
        ("Learn", "Ask course questions and receive grounded explanations with citations."),
        ("Practice", "Generate quizzes, answer them, and review explanations."),
        ("Improve", "Submit answers for grading feedback and recommended review topics."),
    ]
    for col, (title, body) in zip(student_cols, student_cards):
        with col:
            st.markdown(f"<div class='student-card'><h4>{title}</h4>{body}</div>", unsafe_allow_html=True)

    st.markdown("### Required Project Components")
    st.dataframe(component_status(), width="stretch", hide_index=True)
    st.markdown("<span class='success-pill'>All required system components are implemented</span>", unsafe_allow_html=True)


def show_agents_prompts_tab() -> None:
    st.subheader("Agents, LangGraph, and Prompt Templates")
    st.write(
        "This tab shows the concrete implementation behind the student experience: explicit graph nodes, "
        "routing edges, and prompt templates for each educational behavior."
    )

    blueprint = get_graph_blueprint()
    if blueprint["langgraph_available"]:
        st.success("Real LangGraph execution is available and used by `run_genai_mentor`.")
    else:
        st.warning(
            "LangGraph is declared in `requirements.txt` but is not installed in this Python environment. "
            "The same graph nodes run through the local sequential graph fallback until dependencies are installed."
        )
        if blueprint.get("import_error"):
            st.caption(f"Import status: {blueprint['import_error']}")

    st.markdown("### Agent Workflow")
    st.caption("Static graph topology. Orange highlighting appears in the Trace tab after a response runs.")
    render_agent_graph()
    st.markdown(
        " ".join(
            f"<span class='agent-chip'>{node['agent']}</span>"
            for node in blueprint["nodes"]
        ),
        unsafe_allow_html=True,
    )
    st.dataframe(blueprint["nodes"], width="stretch", hide_index=True)

    st.markdown("### Graph Edges")
    edge_rows = [
        {"from": edge[0], "to": edge[1], "condition": edge[2] if len(edge) > 2 else "always"}
        for edge in blueprint["edges"]
    ]
    st.dataframe(edge_rows, width="stretch", hide_index=True)

    st.markdown("### Active Prompt Templates")
    prompt_rows = [
        {
            "name": template.name,
            "purpose": template.purpose,
            "required_inputs": ", ".join(template.required_inputs) or "none",
        }
        for template in list_prompt_templates()
    ]
    st.dataframe(prompt_rows, width="stretch", hide_index=True)

    with st.expander("Read the exact prompt templates"):
        for template in list_prompt_templates():
            st.markdown(f"**{template.name}** — {template.purpose}")
            st.code(template.template.strip(), language="text")


def rag_pipeline_dot() -> str:
    return """
digraph RAGPipeline {
  rankdir=TB;
  graph [bgcolor="transparent", pad="0.4", nodesep="0.5", ranksep="0.7", fontname="Helvetica"];
  node  [fontname="Helvetica", fontsize=10, margin="0.15,0.10"];
  edge  [fontname="Helvetica", fontsize=9,  color="#94a3b8", arrowsize=0.8];

  query  [label="User Query",          shape=oval,    style="filled", fillcolor="#dbeafe", color="#3b82f6", penwidth=2];
  expand [label="Query Expansion\\nenriched_query\\n(clean + domain synonyms)", shape=box, style="rounded,filled", fillcolor="#ede9fe", color="#7c3aed", penwidth=1.6];
  router [label="Mode Router",         shape=diamond, style="filled", fillcolor="#fef9c3", color="#ca8a04", penwidth=1.8];

  subgraph cluster_offline {
    label="Offline Retrieval";
    style="rounded";
    color="#bbf7d0";
    bgcolor="#f0fdf4";
    fontsize=10;
    bm25   [label="BM25\\n(keyword)",   shape=box, style="rounded,filled", fillcolor="#dcfce7", color="#16a34a"];
    chroma [label="ChromaDB\\n(semantic)", shape=box, style="rounded,filled", fillcolor="#dcfce7", color="#16a34a"];
  }

  subgraph cluster_online {
    label="MultiSourceOnlineRetriever  (7 providers)";
    style="rounded";
    color="#c7d2fe";
    bgcolor="#eef2ff";
    fontsize=10;
    ddg  [label="DuckDuckGo", shape=box, style="rounded,filled", fillcolor="#e0f2fe", color="#0284c7"];
    wiki [label="Wikipedia",  shape=box, style="rounded,filled", fillcolor="#e0f2fe", color="#0284c7"];
    arxv [label="arXiv",      shape=box, style="rounded,filled", fillcolor="#e0f2fe", color="#0284c7"];
    s2   [label="Semantic Scholar", shape=box, style="rounded,filled", fillcolor="#e0f2fe", color="#0284c7"];
    gh   [label="GitHub",     shape=box, style="rounded,filled", fillcolor="#e0f2fe", color="#0284c7"];
    se   [label="StackExchange", shape=box, style="rounded,filled", fillcolor="#e0f2fe", color="#0284c7"];
    yt   [label="YouTube *",  shape=box, style="rounded,filled", fillcolor="#e0f2fe", color="#0284c7"];
  }

  merge   [label="Merge Candidates",                           shape=box, style="rounded,filled", fillcolor="#fef3c7", color="#d97706", penwidth=1.6];
  rerank  [label="4-Signal Reranker\\nRelevance · Keyword · Metadata · Authority", shape=box, style="rounded,filled", fillcolor="#ffe4e6", color="#e11d48", penwidth=1.6];
  ce      [label="Cross-Encoder\\n(sentence-transformers, optional)", shape=box, style="rounded,filled", fillcolor="#fce7f3", color="#db2777"];
  results [label="RetrievedChunk List\\n(ranked, top-k)",      shape=oval, style="filled", fillcolor="#dcfce7", color="#16a34a", penwidth=2];

  query  -> expand;
  expand -> router;

  router -> bm25   [label=" offline / hybrid"];
  router -> chroma [label=" offline / hybrid"];
  router -> ddg    [label=" online / hybrid"];
  router -> wiki;
  router -> arxv;
  router -> s2;
  router -> gh;
  router -> se;
  router -> yt;

  bm25   -> merge;
  chroma -> merge;
  ddg    -> merge;
  wiki   -> merge;
  arxv   -> merge;
  s2     -> merge;
  gh     -> merge;
  se     -> merge;
  yt     -> merge;

  merge  -> rerank;
  ce     -> rerank [label=" optional", style=dashed, color="#db2777"];
  rerank -> results;
}
"""


def show_rag_tab() -> None:
    st.subheader("RAG Inspector")
    st.write("Use this tab to prove where the answer evidence comes from before the tutor writes a response.")

    with st.expander("Pipeline Architecture", expanded=False):
        st.caption(
            "Query expansion (mix) enriches the search string with domain synonyms before retrieval. "
            "Offline uses BM25 + ChromaDB on course chunks; online queries 7 live sources. "
            "The 4-signal reranker blends relevance, keyword BM25, metadata, and domain authority. "
            "* YouTube requires YOUTUBE_API_KEY."
        )
        try:
            st.graphviz_chart(rag_pipeline_dot())
        except Exception:
            st.code(rag_pipeline_dot(), language="dot")
    query = st.text_input("Retrieval query", "Explain hybrid search in RAG based on our course lectures.")
    mode = st.radio(
        "Retrieval mode to inspect",
        ["offline_only", "hybrid", "online_only"],
        index=0,
        key="rag_mode",
        horizontal=True,
    )
    st.caption("Offline uses course PDFs. Hybrid adds approved online sources when configured. Online-only is for current external facts.")
    if st.button("Run Retrieval", type="primary", width="stretch"):
        with st.spinner("Retrieving sources..."):
            retriever = HybridRetriever()
            chunks = retriever.retrieve(query, mode=mode)
            diagnostics = retriever.last_status
        if diagnostics.get("online"):
            online_status = diagnostics["online"]
            with st.expander("Online retrieval diagnostics", expanded=not chunks and mode in {"online_only", "hybrid"}):
                st.json(online_status)
                if online_status.get("message"):
                    st.caption(online_status["message"])
                if any(provider.get("name") == "tavily" and provider.get("status") == "skipped" for provider in online_status.get("providers", [])):
                    st.info("For the most reliable online retrieval, add `TAVILY_API_KEY` to `.env`. The app also tries the maintained `ddgs` package as a no-key fallback.")
        if chunks:
            st.success(f"Retrieved {len(chunks)} ranked evidence chunks.")
            rows = [
                {
                    "source": item.chunk.source,
                    "page": item.chunk.page,
                    "topic": item.chunk.topic,
                    "semantic": item.semantic_score,
                    "keyword": item.keyword_score,
                    "final": item.final_score,
                    "preview": item.chunk.text[:240],
                }
                for item in chunks
            ]
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.warning("No chunks returned. Build indexes or use offline data before final demo.")


def show_trace_tab() -> None:
    st.subheader("Latest Agent Trace")
    st.write("Every chat run saves a trace so you can inspect routing, tools, retrieved chunks, checker feedback, and final answer.")
    source_label, payload = latest_execution_payload()
    if payload:
        execution_path = payload.get("execution_path") or fallback_execution_path_from_trace(payload)
        st.markdown("### Agent Graph for Latest Response")
        st.caption(f"Source: `{source_label}`")
        render_agent_graph(execution_path)

        st.markdown("### Agents/Tools That Ran Before Delivery")
        render_execution_path(execution_path)

        decision = payload.get("router_decision", {})
        metric_cols = st.columns(4)
        metric_cols[0].metric("Route", decision.get("retrieval_mode", "unknown"))
        metric_cols[1].metric("Intent", str(decision.get("intent", "unknown"))[:24])
        metric_cols[2].metric("Retrieved chunks", len(payload.get("retrieved_content", payload.get("retrieved_chunks", []))))
        metric_cols[3].metric("Tool calls", len(payload.get("tool_calls", [])))
    else:
        st.info("No active response yet. Ask a student question or run the backend demo chat first.")

    latest = latest_file(TRACE_DIR, "trace_*.json")
    if latest is None:
        st.info("No saved traces yet. Ask a question in Chat Tutor first.")
        return
    st.caption(str(latest))
    try:
        st.markdown("### Raw Saved Trace")
        st.json(json.loads(latest.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        st.code(latest.read_text(encoding="utf-8"), language="json")


def show_finetuning_tab() -> None:
    st.subheader("Fine-Tuning Evidence")
    st.info("Fine-tuning shapes the tutor/examiner/critic behavior. Course facts still come from RAG citations.")
    counts = finetune_counts()
    if counts:
        st.dataframe(counts, width="stretch", hide_index=True)
    else:
        st.warning("No fine-tuning datasets found.")

    output_files = list(FINAL_ADAPTER_DIR.glob("**/*")) if FINAL_ADAPTER_DIR.exists() else []
    training_log = OUTPUTS_DIR / "finetune" / "training_log.json"
    training_status = "not run"
    if training_log.exists():
        try:
            training_data = json.loads(training_log.read_text(encoding="utf-8"))
            training_status = training_data.get("status", "unknown")
        except json.JSONDecodeError:
            training_data = {}
            training_status = "unknown"
    else:
        training_data = {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Final adapter files", len([item for item in output_files if item.is_file()]))
    col2.metric("Base model", FINETUNE_BASE_MODEL)
    col3.metric("GPU training", "Completed" if training_status == "completed" else "Pending")

    if training_status == "completed":
        st.success("LoRA adapter training completed and adapter artifacts are available.")
    else:
        st.markdown("**Priority to finish:** train LoRA on GPU/MPS, save adapter artifacts, then run base-vs-tuned comparison.")

    with st.expander("What was trained and where it lives", expanded=True):
        st.markdown(
            """
            - **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
            - **Device:** Apple MPS
            - **Split:** 800 train / 100 validation / 100 test
            - **Final adapter:** `outputs/finetune/qwen_0_5b_lora_adapter_salma/`
            - **Evaluation:** `outputs/finetune/results/evaluation_summary.json`
            """
        )

    adapter_options = [option for option in list_chat_model_options(include_unavailable=True) if option.kind == "lora_adapter"]
    if adapter_options:
        with st.expander("Saved adapter inventory", expanded=True):
            st.dataframe(
                [
                    {
                        "Model name": option.label.split(": ", 1)[-1],
                        "Owner suffix": "Khadija" if option.label.endswith("_khadija") else "Fatma" if option.label.endswith("_fatma") else "Salma" if option.label.endswith("_salma") else "Other",
                        "Base model": option.base_model,
                        "Path": option.path,
                        "Load status": option.status,
                    }
                    for option in adapter_options
                ],
                width="stretch",
                hide_index=True,
            )

    with st.expander("Implementation files"):
        st.code(
            "\n".join([
                "src/finetuning/prepare_dataset.py",
                "src/finetuning/train_lora.py",
                "src/finetuning/inference_lora.py",
                "scripts/03_generate_finetune_data.py",
                "scripts/04_train_lora.py",
                "notebooks/03_finetuning_complete.ipynb",
            ]),
            language="text",
        )

    with st.expander("Optional smoke-test command"):
        st.code(
            "FINETUNE_OUTPUT_ADAPTER_DIR=run_check_smoke_adapter FINETUNE_MAX_TRAIN_EXAMPLES=32 FINETUNE_MAX_EVAL_EXAMPLES=8 FINETUNE_MAX_LENGTH=512 FINETUNE_EPOCHS=1 python3 scripts/04_train_lora.py",
            language="bash",
        )

    if training_data:
        with st.expander("Training log"):
            st.json(training_data)

    adapter_paths = [str(item.relative_to(ROOT_DIR)) for item in sorted((OUTPUTS_DIR / "finetune").glob("**/*")) if item.is_file()]
    if adapter_paths:
        with st.expander("Fine-tuning output files"):
            st.dataframe([{"file": path} for path in adapter_paths], width="stretch", hide_index=True)

    sample_path = DATA_DIR / "finetuning/sample_review.md"
    if sample_path.exists():
        with st.expander("Dataset quality review"):
            st.markdown(sample_path.read_text(encoding="utf-8")[:5000])


def show_evaluation_tab() -> None:
    st.subheader("Evaluation Dashboard")
    st.write("This tab summarizes whether the system retrieves evidence, answers clearly, and blocks unsafe requests.")
    summary = OUTPUTS_DIR / "evaluation/evaluation_summary.md"
    results = OUTPUTS_DIR / "evaluation/evaluation_results.csv"
    if summary.exists():
        st.markdown(summary.read_text(encoding="utf-8"))
    else:
        st.warning("Evaluation report not generated yet. Run `python3 scripts/05_run_evaluation.py` before submission.")
    if results.exists():
        st.dataframe(pd.read_csv(results), width="stretch", hide_index=True)

    st.divider()
    st.markdown("### RAG Ablation Evaluator")
    st.caption("Use this control to turn RAG off for evaluation and compare the same prompt with retrieved evidence enabled.")

    eval_query = st.text_area(
        "Evaluation prompt",
        "Explain RAG and why it reduces hallucinations using course evidence.",
        key="rag_ablation_query",
        height=90,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        eval_mode = st.radio(
            "Evaluation mode",
            ["Compare RAG vs No RAG", "RAG on only", "RAG off only"],
            horizontal=False,
            key="rag_ablation_mode",
        )
    with col2:
        rag_mode = st.selectbox(
            "RAG-on retrieval mode",
            ["offline_only", "hybrid", "online_only"],
            index=0,
            key="rag_ablation_rag_mode",
        )
    with col3:
        eval_level = st.selectbox(
            "Student level",
            list(STUDENT_LEVELS.keys()),
            index=1,
            format_func=lambda value: STUDENT_LEVELS[value],
            key="rag_ablation_student_level",
        )

    eval_model_options = student_chat_model_options()
    eval_model_ids = [option.id for option in eval_model_options]
    eval_recommended = get_recommended_chat_model_id()
    eval_model_index = eval_model_ids.index(eval_recommended) if eval_recommended in eval_model_ids else 0
    eval_model_id = st.selectbox(
        "Evaluation response model",
        eval_model_ids,
        index=eval_model_index,
        format_func=student_chat_model_label,
        key="rag_ablation_model",
    )

    if st.button("Run RAG ablation evaluation", type="primary", width="stretch"):
        if eval_mode == "Compare RAG vs No RAG":
            planned_runs = [("RAG on", rag_mode), ("RAG off", "no_retrieval")]
        elif eval_mode == "RAG on only":
            planned_runs = [("RAG on", rag_mode)]
        else:
            planned_runs = [("RAG off", "no_retrieval")]

        ablation_results = []
        with st.spinner("Running evaluation prompt through selected routing modes..."):
            for label, override in planned_runs:
                result = run_genai_mentor(
                    eval_query,
                    ui_options={
                        "retrieval_override": override,
                        "student_level": eval_level,
                        "chat_model_id": eval_model_id,
                        "n_questions": 3,
                    },
                )
                ablation_results.append({"label": label, "override": override, "result": result})
        st.session_state.rag_ablation_results = ablation_results

    ablation_results = st.session_state.get("rag_ablation_results", [])
    if ablation_results:
        summary_rows = []
        for item in ablation_results:
            result = item["result"]
            checker = result.get("checker_feedback", {})
            summary_rows.append(
                {
                    "run": item["label"],
                    "override": item["override"],
                    "actual_route": result.get("router_decision", {}).get("retrieval_mode"),
                    "retrieved_chunks": len(result.get("retrieved_content", [])),
                    "answer_chars": len(result.get("answer", "")),
                    "grounded": checker.get("grounded", "n/a"),
                    "citations": checker.get("citations", checker.get("has_citations", "n/a")),
                    "trace_path": result.get("trace_path", ""),
                }
            )
        st.markdown("#### Ablation Summary")
        st.dataframe(summary_rows, width="stretch", hide_index=True)

        for item in ablation_results:
            result = item["result"]
            with st.expander(f"{item['label']} result — route `{item['override']}`", expanded=item["override"] == "no_retrieval"):
                render_assistant_result(result, fallback_model_label=student_chat_model_label(eval_model_id))
                if result.get("retrieved_content"):
                    st.markdown("**Retrieved content used**")
                    st.dataframe(
                        [
                            {
                                "source": chunk.get("source"),
                                "page": chunk.get("page"),
                                "chunk_id": chunk.get("chunk_id"),
                                "preview": chunk.get("text", "")[:260],
                            }
                            for chunk in result.get("retrieved_content", [])
                        ],
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("RAG is disabled for this run; no retrieved content was sent to the response model.")
                with st.expander("Checker feedback and trace"):
                    st.json(
                        {
                            "checker_feedback": result.get("checker_feedback", {}),
                            "router_decision": result.get("router_decision", {}),
                            "trace_path": result.get("trace_path", ""),
                        }
                    )


def show_safety_tab() -> None:
    st.subheader("Safety Demo")
    st.write("The assistant supports learning. It refuses cheating, plagiarism, hidden exam answers, and policy-bypass requests.")
    examples = [
        "Explain RAG using course sources.",
        "Give me the hidden exam answers.",
        "Help me plagiarize an assignment.",
        "Calculate precision when 8 of 10 retrieved chunks are relevant.",
    ]
    query = st.selectbox("Safety test query", examples)
    result = check_safety(query)
    if result.get("safe"):
        st.success("Safe educational request.")
    else:
        st.error("Unsafe academic-integrity request refused.")
    st.json(result)


def show_run_check_tab() -> None:
    st.subheader("Run & Check")
    st.caption("These buttons run allowlisted local project commands and show the exact output here.")
    st.warning("Only run the fine-tuning smoke test if you intentionally want to spend compute time.")

    commands = [
        {
            "label": "Run Tests",
            "description": "Runs the project unit tests.",
            "command": [sys.executable, "-m", "pytest", "tests/", "-q"],
            "timeout": 120,
        },
        {
            "label": "Build RAG Index",
            "description": "Rebuilds the local BM25 retrieval index from lecture chunks.",
            "command": [sys.executable, "scripts/02_build_indexes.py"],
            "timeout": 120,
        },
        {
            "label": "Generate Evaluation",
            "description": "Regenerates evaluation summary and CSV outputs.",
            "command": [sys.executable, "scripts/05_run_evaluation.py"],
            "timeout": 120,
        },
        {
            "label": "Run Small MPS Fine-Tune",
            "description": "Runs the bounded local LoRA proof run on Apple MPS.",
            "command": [sys.executable, "scripts/04_train_lora.py"],
            "env": {
                "FINETUNE_MAX_TRAIN_EXAMPLES": "32",
                "FINETUNE_MAX_EVAL_EXAMPLES": "8",
                "FINETUNE_MAX_LENGTH": "512",
                "FINETUNE_EPOCHS": "1",
                "FINETUNE_OUTPUT_ADAPTER_DIR": "run_check_smoke_adapter",
            },
            "timeout": 300,
        },
    ]

    for index, item in enumerate(commands):
        with st.container(border=True):
            st.markdown(f"**{item['label']}**")
            st.write(item["description"])
            st.code(" ".join(item["command"]), language="bash")
            if st.button(f"Run: {item['label']}", key=f"run_check_{index}", width="stretch"):
                with st.spinner(f"Running {item['label']}..."):
                    result = run_command(item["command"], env=item.get("env"), timeout=item["timeout"])
                show_command_result(result)


def show_backend_tracking_view() -> None:
    st.subheader("Backend Tracking")
    st.caption("Use this area to inspect implementation evidence, graph routing, retrieval diagnostics, traces, fine-tuning, evaluation, and safety.")

    tabs = st.tabs([
        "🏠 Overview",
        "🧠 Agents & Prompts",
        "🔎 Evidence/RAG",
        "🧭 Agent Trace",
        "🧪 Fine-Tuning",
        "📊 Evaluation",
        "🛡️ Safety",
        "✅ Run & Check",
    ])

    with tabs[0]:
        show_overview_tab()
    with tabs[1]:
        show_agents_prompts_tab()
    with tabs[2]:
        show_rag_tab()
    with tabs[3]:
        show_trace_tab()
    with tabs[4]:
        show_finetuning_tab()
    with tabs[5]:
        show_evaluation_tab()
    with tabs[6]:
        show_safety_tab()
    with tabs[7]:
        show_run_check_tab()


show_hero()
show_mode_selector()

if st.session_state.view_mode == "student":
    show_student_view()
else:
    show_backend_tracking_view()
