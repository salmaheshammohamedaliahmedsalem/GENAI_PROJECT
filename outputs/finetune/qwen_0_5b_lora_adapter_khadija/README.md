---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
---

# Khadija Qwen 0.5B LoRA Adapter

This folder packages Khadija's LoRA adapter weights for use inside the GenAI Mentor response model selector.

## Adapter

- **Owner suffix:** `khadija`
- **Adapter path:** `outputs/finetune/qwen_0_5b_lora_adapter_khadija/`
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter type:** PEFT/LoRA
- **LoRA config:** `r=8`, `lora_alpha=16`, `lora_dropout=0.05`
- **Target modules:** `k_proj`, `q_proj`, `down_proj`, `gate_proj`, `o_proj`, `up_proj`, `v_proj`

## Important Notes

- This is an adapter, not a standalone model.
- It must be loaded on top of `Qwen/Qwen2.5-0.5B-Instruct`.
- The adapter weights came from `/Users/salmaheshamsalem/Desktop/adapter_model.safetensors`.
- Tokenizer/config files are packaged from the matching Qwen 0.5B LoRA adapter format so the app can discover and load the adapter consistently.
