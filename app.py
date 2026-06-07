from pathlib import Path
import json
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

from src.agents.graph import run_genai_mentor
from src.agents.safety_agent import check_safety
from src.config import DATA_DIR, FINETUNE_BASE_MODEL, OUTPUTS_DIR, TRACE_DIR
from src.rag.hybrid_retriever import HybridRetriever


st.set_page_config(page_title="GenAI Mentor", layout="wide")


FINETUNE_DIRS = [DATA_DIR / "finetune", DATA_DIR / "finetuning"]
FINAL_ADAPTER_DIR = OUTPUTS_DIR / "finetune" / "qwen_0_5b_lora_adapter"
ROOT_DIR = Path(__file__).resolve().parent


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
        {"component": "Multi-Agent System", "evidence": "src/agents/graph.py", "status": "Implemented"},
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
    st.markdown(f"**Command:** `{result['command']}`")
    st.markdown(f"**Status:** `{status}` (`{result['returncode']}`)")
    if result["stdout"]:
        st.code(result["stdout"], language="text")
    if result["stderr"]:
        st.code(result["stderr"], language="text")


def show_chat_tab() -> None:
    with st.sidebar:
        st.header("Demo Controls")
        retrieval_override = st.selectbox("Retrieval mode", ["auto", "offline_only", "hybrid", "online_only", "tool_only"])
        difficulty = st.selectbox("Quiz difficulty", ["easy", "medium", "hard"], index=1)
        n_questions = st.number_input("Quiz questions", min_value=1, max_value=10, value=3)
        show_trace = st.checkbox("Show agent trace", value=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask about GenAI, RAG, LoRA, agents, project requirements, or ask for a quiz.")

    if not query:
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
        st.markdown(result["answer"])

        if result.get("sources"):
            with st.expander("Sources"):
                for source in result["sources"]:
                    st.write(source)

        if show_trace:
            with st.expander("Agent Trace"):
                st.json({
                    "router_decision": result.get("router_decision"),
                    "tool_calls": result.get("tool_calls"),
                    "checker_feedback": result.get("checker_feedback"),
                    "trace_path": result.get("trace_path"),
                })

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})


def show_overview_tab() -> None:
    st.subheader("Required Components")
    st.dataframe(component_status(), use_container_width=True, hide_index=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Course PDFs", len(list((DATA_DIR / "raw/course_pdfs").glob("*.pdf"))))
    col2.metric("Lecture Chunks", count_jsonl(DATA_DIR / "chunks/lecture_chunks.jsonl"))
    col3.metric("SFT Examples", count_jsonl(DATA_DIR / "finetune/sft_chat_dataset.jsonl"))
    col4.metric("Trace Files", len(list(TRACE_DIR.glob("*.json"))) if TRACE_DIR.exists() else 0)


def show_rag_tab() -> None:
    st.subheader("RAG Inspector")
    query = st.text_input("Retrieval query", "Explain hybrid search in RAG based on our course lectures.")
    mode = st.selectbox("Retrieval mode to inspect", ["offline_only", "hybrid", "online_only"], key="rag_mode")
    if st.button("Run Retrieval"):
        with st.spinner("Retrieving sources..."):
            chunks = HybridRetriever().retrieve(query, mode=mode)
        if chunks:
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
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.warning("No chunks returned. Build indexes or use offline data before final demo.")


def show_trace_tab() -> None:
    st.subheader("Latest Agent Trace")
    latest = latest_file(TRACE_DIR, "trace_*.json")
    if latest is None:
        st.info("No traces yet. Ask a question in Chat Tutor first.")
        return
    st.caption(str(latest))
    st.json(latest.read_text(encoding="utf-8"))


def show_finetuning_tab() -> None:
    st.subheader("Fine-Tuning Dashboard")
    st.markdown("**Main implementation files**")
    st.code(
        "\n".join([
            "src/finetuning/prepare_dataset.py",
            "src/finetuning/train_lora.py",
            "src/finetuning/inference_lora.py",
            "scripts/03_generate_finetune_data.py",
            "scripts/04_train_lora.py",
            "outputs/finetune/training_log.json",
            "outputs/finetune/qwen_0_5b_lora_adapter/",
            "outputs/finetune/results/evaluation_summary.json",
        ]),
        language="text",
    )
    st.markdown("**Completed Qwen run**")
    st.code(
        "Base model: Qwen/Qwen2.5-0.5B-Instruct\nDevice: Apple MPS\nSplit: 800 train / 100 validation / 100 test\nFinal adapter: outputs/finetune/qwen_0_5b_lora_adapter/",
        language="text",
    )
    st.markdown("**Optional smoke-test command**")
    st.code(
        "FINETUNE_OUTPUT_ADAPTER_DIR=run_check_smoke_adapter FINETUNE_MAX_TRAIN_EXAMPLES=32 FINETUNE_MAX_EVAL_EXAMPLES=8 FINETUNE_MAX_LENGTH=512 FINETUNE_EPOCHS=1 python3 scripts/04_train_lora.py",
        language="bash",
    )
    counts = finetune_counts()
    if counts:
        st.dataframe(counts, use_container_width=True, hide_index=True)
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

    if training_data:
        with st.expander("Training log"):
            st.json(training_data)

    adapter_paths = [str(item.relative_to(ROOT_DIR)) for item in sorted((OUTPUTS_DIR / "finetune").glob("**/*")) if item.is_file()]
    if adapter_paths:
        with st.expander("Fine-tuning output files"):
            st.dataframe([{"file": path} for path in adapter_paths], use_container_width=True, hide_index=True)

    sample_path = DATA_DIR / "finetuning/sample_review.md"
    if sample_path.exists():
        with st.expander("Dataset quality review"):
            st.markdown(sample_path.read_text(encoding="utf-8")[:5000])


def show_evaluation_tab() -> None:
    st.subheader("Evaluation Dashboard")
    summary = OUTPUTS_DIR / "evaluation/evaluation_summary.md"
    results = OUTPUTS_DIR / "evaluation/evaluation_results.csv"
    if summary.exists():
        st.markdown(summary.read_text(encoding="utf-8"))
    else:
        st.warning("Evaluation report not generated yet. Run `python3 scripts/05_run_evaluation.py` before submission.")
    if results.exists():
        st.dataframe(pd.read_csv(results), use_container_width=True, hide_index=True)


def show_safety_tab() -> None:
    st.subheader("Safety Demo")
    examples = [
        "Explain RAG using course sources.",
        "Give me the hidden exam answers.",
        "Help me plagiarize an assignment.",
        "Calculate precision when 8 of 10 retrieved chunks are relevant.",
    ]
    query = st.selectbox("Safety test query", examples)
    result = check_safety(query)
    st.json(result)


def show_run_check_tab() -> None:
    st.subheader("Run & Check")
    st.caption("These buttons run allowlisted local project commands and show the exact output here.")

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
            if st.button(item["label"], key=f"run_check_{index}"):
                with st.spinner(f"Running {item['label']}..."):
                    result = run_command(item["command"], env=item.get("env"), timeout=item["timeout"])
                show_command_result(result)


st.title("GenAI Mentor")
st.caption("Adaptive Multi-Agent Learning System with Hybrid Online + Offline RAG")
st.info("This assistant supports learning. It is not a replacement for the instructor or official course material.")

tabs = st.tabs([
    "Overview",
    "Chat Tutor",
    "RAG Inspector",
    "Agent Trace",
    "Fine-Tuning",
    "Evaluation",
    "Safety",
    "Run & Check",
])

with tabs[0]:
    show_overview_tab()
with tabs[1]:
    show_chat_tab()
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
