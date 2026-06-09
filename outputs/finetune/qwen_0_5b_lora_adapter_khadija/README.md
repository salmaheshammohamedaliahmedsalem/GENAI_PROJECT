---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
---

# Khadija Level-Adaptive Tutor LoRA Adapter

This folder packages Khadija's corrected level-adaptive tutor LoRA adapter for the GenAI Mentor response model selector.

## Source

- **Original local folder:** `/Users/salmaheshamsalem/Desktop/level_adaptive_tutor_final_adapter copy`
- **Packaged project path:** `outputs/finetune/qwen_0_5b_lora_adapter_khadija/`
- **Owner suffix:** `_khadija`

## Adapter Details

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter type:** PEFT/LoRA
- **Task type:** `CAUSAL_LM`
- **LoRA rank:** `8`
- **LoRA alpha:** `16`
- **LoRA dropout:** `0.05`
- **Target modules:** `o_proj`, `v_proj`, `q_proj`, `up_proj`, `k_proj`, `gate_proj`, `down_proj`

## Included Files

- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`

The zipped duplicate from the source folder is intentionally not included because the unzipped adapter weights are already tracked.

## Intended Use

- Appears as `Khadija fine-tuned model (Qwen 0.5B LoRA)` in the Student response model menu.
- Can be inspected in Backend Tracking with the rest of the saved adapter inventory.
- Should be loaded on top of `Qwen/Qwen2.5-0.5B-Instruct`; it is not a standalone model.

## Runtime Notes

Local LoRA inference requires the dependencies in `requirements_finetune.txt` and access to the base model. On Streamlit Community Cloud, the adapter can be listed as an artifact, but local LoRA execution may be unavailable unless the deployment includes the heavy fine-tuning dependencies and model cache.
