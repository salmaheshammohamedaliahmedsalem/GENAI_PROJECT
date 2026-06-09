---
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- peft
- education
- mps-smoke-test
---

# GenAI Mentor TinyLlama LoRA Adapter

This adapter is a bounded local MPS training artifact from `scripts/04_train_lora.py`. It proves the project’s PEFT training path works end-to-end on Apple Silicon.

## Training Run

- **Base model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Adapter path:** `outputs/finetune/lora_adapter/`
- **Dataset split:** `data/finetune/train.jsonl` and `data/finetune/val.jsonl`
- **Examples used:** 32 train / 8 validation
- **Hardware:** Apple Silicon using PyTorch MPS
- **Epochs:** 1
- **Max sequence length:** 512

## Project Role

The primary final adapter is `outputs/finetune/qwen_0_5b_lora_adapter/`. This TinyLlama adapter is retained as a small reproducibility artifact for the script-based training workflow.

## Limitations

- This is an adapter, not a standalone model.
- It is a short smoke-test run and is not the strongest final model.
- Use the Qwen adapter for the report-ready fine-tuning demonstration.
