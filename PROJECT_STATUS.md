# GenAI Mentor - Current Project Status

**Last updated:** June 6, 2026  
**Status:** Working demo with MPS LoRA adapter training evidence

## Executive Summary

The project now has a working Streamlit GUI, passing tests, prepared fine-tuning splits, generated evaluation outputs, local offline retrieval from the lecture chunks, and a completed bounded LoRA adapter training run on Apple MPS.

## Current Verification

| Check | Result |
| --- | --- |
| Unit tests | `7 passed, 1 warning` with `python3 -m pytest tests/ -q` |
| Syntax check | Passed for app, scripts, fine-tuning, and retrieval modules |
| Streamlit GUI | Running at `http://localhost:8501` |
| RAG index | BM25 index built from `data/chunks/lecture_chunks.jsonl` |
| Evaluation outputs | Generated in `outputs/evaluation/` |
| Fine-tuning splits | Generated in `data/finetune/` |
| LoRA adapter | Completed on Apple MPS |

## Required Components

| Requirement | Evidence | Status |
| --- | --- | --- |
| Prompt Design | `src/llm/prompts.py` | Implemented |
| RAG | `src/rag/`, `data/processed/bm25_index.pkl` | Implemented locally with BM25 fallback; semantic Chroma is opt-in via `ENABLE_SEMANTIC_RAG=true` |
| Fine-tuning / PEFT | `src/finetuning/`, `data/finetune/*.jsonl`, `outputs/finetune/lora_adapter/` | MPS adapter training completed |
| Tools / Function Calling | `src/tools/` | Implemented |
| Multi-Agent Setup | `src/agents/graph.py` | Implemented |
| Evaluation | `src/evaluation/`, `outputs/evaluation/` | Implemented and generated |
| Ethics / Safety | `src/agents/safety_agent.py`, `docs/ethics_safety.md` | Implemented |
| GUI | `app.py` | Implemented with full showcase tabs |

## Fine-Tuning Status

Prepared chat-format SFT data:

| Split | Examples |
| --- | ---: |
| `data/finetune/train.jsonl` | 4,780 |
| `data/finetune/val.jsonl` | 598 |
| `data/finetune/test.jsonl` | 598 |
| `data/finetune/sft_chat_dataset.jsonl` | 5,976 |

Training command:

```bash
python3 scripts/04_train_lora.py
```

Current local result:

```json
{
  "status": "completed",
  "device": "mps",
  "train_examples": 32,
  "validation_examples": 8,
  "epochs": 1.0,
  "max_length": 512
}
```

Adapter artifacts are saved in `outputs/finetune/lora_adapter/`. This was a bounded local MPS run for project evidence. For a stronger final experiment, increase `FINETUNE_MAX_TRAIN_EXAMPLES` or omit it and rerun on MPS/GPU.

## Final GUI

Run:

```bash
streamlit run app.py
```

The GUI now includes:

- **Overview:** required component checklist and live data metrics.
- **Chat Tutor:** multi-agent chat with retrieval controls, quiz controls, sources, and trace.
- **RAG Inspector:** retrieval query runner with chunk/source scoring table.
- **Agent Trace:** latest saved agent trace JSON.
- **Fine-Tuning:** dataset counts, adapter status, training log status, and dataset quality review.
- **Evaluation:** generated evaluation summary and result table.
- **Safety:** refusal and safety classification demo.

## Recommended Submission Order

1. Demo the GUI at `http://localhost:8501`.
2. Show `Overview`, `RAG Inspector`, `Fine-Tuning`, `Evaluation`, and `Safety` tabs.
3. Show the `Fine-Tuning` tab with completed MPS training status and adapter artifacts.
4. If more time is available, rerun LoRA with more examples for stronger quality evidence.
