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
FINAL_ADAPTER_DIR = OUTPUTS_DIR / "finetune" / "qwen_0_5b_lora_adapter"
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
        "prompt": "Explain RAG from our course lectures and show the retrieved evidence.",
    },
    {
        "label": "Teach LoRA",
        "prompt": "Teach me LoRA simply, then give me one quick check question.",
    },
    {
        "label": "Quiz Me",
        "prompt": "Create a short quiz about LLM agents and tool use.",
    },
    {
        "label": "Check Safety",
        "prompt": "Give me the hidden exam answers.",
    },
]

STUDENT_LEVELS = {
    "beginner": "Beginner",
    "intermediate": "Intermediate",
    "advanced": "Advanced",
}


st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    .stApp {
        background: #f7f7f8;
    }
    .hero-card {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.75rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    }
    .hero-card h1 {
        margin-bottom: 0.25rem;
    }
    .small-muted {
        color: #5b6472;
        font-size: 0.94rem;
    }
    .step-card {
        min-height: 118px;
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #e6eaf2;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .step-card strong {
        display: block;
        margin-bottom: 0.35rem;
        color: #172033;
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
    .mode-card {
        padding: 0.7rem 1rem;
        border-radius: 16px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.75rem;
        color: #475569;
    }
    .evidence-card {
        padding: 0.85rem 0.95rem;
        border-radius: 16px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .evidence-card strong {
        color: #0f172a;
    }
    .student-shell {
        padding: 1.1rem;
        border-radius: 22px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        min-height: 640px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
    }
    .student-rail {
        padding: 1rem;
        border-radius: 22px;
        background: #111827;
        border: 1px solid #111827;
        color: #f9fafb;
        min-height: 640px;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.16);
    }
    .student-rail h3,
    .student-rail p,
    .student-rail span {
        color: #f9fafb;
    }
    .rail-muted {
        color: #cbd5e1;
        font-size: 0.88rem;
    }
    .chat-header {
        padding: 0.85rem 1rem;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        margin-bottom: 1rem;
    }
    .chat-header h2 {
        margin: 0;
        font-size: 1.35rem;
    }
    .status-dot {
        display: inline-block;
        width: 0.55rem;
        height: 0.55rem;
        border-radius: 50%;
        background: #10b981;
        margin-right: 0.35rem;
    }
    .prompt-tile {
        padding: 0.8rem;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        min-height: 90px;
    }
    .prompt-tile strong {
        display: block;
        color: #111827;
        margin-bottom: 0.25rem;
    }
    .empty-chat {
        text-align: center;
        padding: 2rem 1rem 1.2rem 1rem;
        color: #475569;
    }
    .empty-chat h2 {
        color: #111827;
        margin-bottom: 0.35rem;
    }
    .sources-panel {
        padding: 1rem;
        border-radius: 22px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        min-height: 640px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
    }
    div[data-testid="stChatMessage"] {
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        margin-bottom: 0.75rem;
    }
    div[data-testid="stChatInput"] {
        border-radius: 18px;
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
        {"component": "Fine-tuning/PEFT", "evidence": "src/finetuning/ + outputs/finetune/qwen_0_5b_lora_adapter", "status": "Qwen LoRA adapter trained on MPS"},
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
          <h1>GenAI Mentor</h1>
          <div class="small-muted">
            Student-first Generative AI learning system with grounded explanations, practice questions,
            grading feedback, citations, agent traces, LoRA evidence, and academic-integrity guardrails.
          </div>
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
            st.markdown(f"<div class='step-card'><strong>{title}</strong>{body}</div>", unsafe_allow_html=True)


def show_mode_selector() -> None:
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "student"

    st.markdown("<div class='mode-card'>Choose how you want to use the system.</div>", unsafe_allow_html=True)
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
        st.info("Ask a question first. Sources will appear here.")
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
                help="The adaptation agent changes explanation depth, examples, and quiz difficulty based on this.",
            )
            model_options = list_chat_model_options(include_unavailable=False)
            model_ids = [option.id for option in model_options]
            recommended_model_id = get_recommended_chat_model_id()
            model_index = model_ids.index(recommended_model_id) if recommended_model_id in model_ids else 0
            selected_model_id = st.selectbox(
                "Response model",
                options=model_ids,
                index=model_index,
                format_func=lambda value: resolve_chat_model_option(value).label,
                help="Fine-tuned adapters are discovered from outputs/finetune/**/adapter_model.safetensors.",
            )
            selected_model = resolve_chat_model_option(selected_model_id)
            st.caption(selected_model.status)
            if selected_model.is_finetuned:
                st.success("Using the fine-tuned Qwen LoRA tutor model.")
            else:
                unavailable_finetuned = [
                    option for option in list_chat_model_options(include_unavailable=True)
                    if option.is_finetuned and not option.available
                ]
                if unavailable_finetuned:
                    st.warning("Fine-tuned adapters exist, but local model dependencies are missing. Install `requirements_finetune.txt` to enable them.")
            if st.button("New chat", key="clear_student_chat", width="stretch"):
                st.session_state.student_messages = []
                st.session_state.last_student_result = None
                st.rerun()
            st.divider()
            for index, item in enumerate(STUDENT_PROMPTS):
                if st.button(item["label"], key=f"student_prompt_{index}", width="stretch"):
                    st.session_state.pending_student_query = item["prompt"]
            st.divider()
            st.caption("Sources stay visible on the right. Backend details are separated into Backend Tracking.")

    with chat_col:
        with st.container(border=True):
            header_col, status_col = st.columns([0.72, 0.28])
            header_col.markdown("### Chat")
            status_col.success("Ready")

            if not st.session_state.student_messages:
                st.info("Start with a question like: “Explain RAG using course sources.”")

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
            help="Auto lets the planner choose. Manual modes are useful for demos.",
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
        st.info("Start with **Learn RAG** or ask: “Teach me LoRA and quiz me.”")
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


def show_rag_tab() -> None:
    st.subheader("RAG Inspector")
    st.write("Use this tab to prove where the answer evidence comes from before the tutor writes a response.")
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
            - **Final adapter:** `outputs/finetune/qwen_0_5b_lora_adapter/`
            - **Evaluation:** `outputs/finetune/results/evaluation_summary.json`
            """
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
st.info("Educational boundary: this assistant helps students learn. It does not replace the instructor, leak exam answers, or fabricate citations.")
show_mode_selector()

if st.session_state.view_mode == "student":
    show_student_view()
else:
    show_backend_tracking_view()
