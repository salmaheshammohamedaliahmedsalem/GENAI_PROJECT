# GenAI Mentor - Current Project Status

**Last updated:** June 7, 2026  
**Status:** Working deployed demo with completed Qwen LoRA adapter training evidence

## Executive Summary

The project has a working Streamlit GUI, passing tests, prepared fine-tuning data, generated evaluation outputs, local offline retrieval from lecture chunks, safety routing, tool use, multi-agent orchestration, and a completed Qwen LoRA adapter training run on Apple MPS.

## Current Verification

| Check | Result |
| --- | --- |
| Unit tests | `16 passed, 1 warning` with `python3 -m pytest tests/ -q` |
| Syntax check | Passed for app, scripts, fine-tuning, and retrieval modules |
| Streamlit GUI | Local: `http://localhost:8501`; deployed: `https://studybuddygenai.streamlit.app/` |
| RAG index | BM25 index built from `data/chunks/lecture_chunks.jsonl` |
| Evaluation outputs | Generated in `outputs/evaluation/` |
| Fine-tuning splits | Generated in `data/finetune/` |
| LoRA adapter | Completed on Apple MPS |

## Required Components

| Requirement | Evidence | Status |
| --- | --- | --- |
| Prompt Design | `src/llm/prompts.py` | Implemented |
| RAG | `src/rag/`, `data/processed/bm25_index.pkl` | Implemented locally with BM25 and approved online retrieval through Tavily or `ddgs` fallback; Chroma semantic retrieval is optional via `requirements_semantic.txt` |
| Fine-tuning / PEFT | `src/finetuning/`, `data/finetune/*.jsonl`, `outputs/finetune/qwen_0_5b_lora_adapter/` | Qwen LoRA adapter training completed on MPS |
| Tools / Function Calling | `src/tools/` | Implemented |
| Multi-Agent Setup | `src/agents/graph.py`, `src/agents/adaptation_agent.py`, `requirements.txt` | Implemented as a LangGraph `StateGraph` with safety, planner, student adaptation, retrieval, response, checker, and trace nodes |
| Evaluation | `src/evaluation/`, `outputs/evaluation/` | Implemented and generated |
| Ethics / Safety | `src/agents/safety_agent.py`, `docs/ethics_safety.md` | Implemented |
| GUI | `app.py` | Implemented with Student mode and Backend Tracking mode |

## Fine-Tuning Status

Prepared chat-format SFT data:

| Split | Examples |
| --- | ---: |
| `data/finetune/train.jsonl` | 800 |
| `data/finetune/val.jsonl` | 100 |
| `data/finetune/test.jsonl` | 100 |
| `data/finetune/sft_chat_dataset.jsonl` | 5,976 |

Training command:

```bash
python3 scripts/04_train_lora.py
```

Current local result:

```json
{
  "status": "completed",
  "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "device": "mps",
  "train_examples": 800,
  "validation_examples": 100,
  "test_examples": 100,
  "epochs": 2.0,
  "train_loss": 2.3253,
  "eval_loss": 2.1984
}
```

Final Qwen adapter artifacts are saved in `outputs/finetune/qwen_0_5b_lora_adapter/`. The smaller `outputs/finetune/lora_adapter/` artifact is retained only as a script-level TinyLlama smoke-test.

## Final GUI

Deployed app:

[StudyBuddy GenAI](https://studybuddygenai.streamlit.app/)

Local run:

```bash
streamlit run app.py
```

The GUI now has two top-level modes:

- **Student:** chat interface with a student-level selector plus a retrieved-content panel showing the exact chunks/sources used for the latest answer.
- **Backend Tracking:** implementation/evidence dashboard with Overview, Agents & Prompts, RAG Inspector, Agent Trace, Fine-Tuning, Evaluation, Safety, and Run & Check tabs.

## Recommended Submission Order

1. Demo the GUI at `http://localhost:8501`.
2. Show **Student** mode with chat and retrieved content.
3. Switch to **Backend Tracking** and show `Overview`, `RAG Inspector`, `Fine-Tuning`, `Evaluation`, and `Safety` tabs.
4. Show the `Fine-Tuning` tab with completed MPS training status and adapter artifacts.
