---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
---

# GenAI Mentor Qwen 0.5B LoRA Adapter

This is the final PEFT/LoRA adapter produced for the GenAI Mentor project. It is trained to reinforce the project’s educational tutor, examiner, and critic response formats while factual course grounding remains handled by the RAG layer.

## Training Run

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter path:** `outputs/finetune/qwen_0_5b_lora_adapter/`
- **Dataset source:** `data/finetune/sft_chat_dataset.jsonl`
- **Examples used:** 1,000 clean examples from the 5,976-example SFT dataset
- **Split:** 800 train / 100 validation / 100 test
- **Hardware:** Apple Silicon using PyTorch MPS
- **Epochs:** 2
- **LoRA config:** `r=8`, `lora_alpha=16`, `lora_dropout=0.05`

## Metrics

- **Train loss:** 2.3253
- **Eval loss:** 2.1984
- **Base vs tuned comparison:** `outputs/finetune/results/base_vs_tuned_comparison.csv`
- **Evaluation summary:** `outputs/finetune/results/evaluation_summary.json`

## Usage

Load this adapter with `src/finetuning/inference_lora.py` or the fine-tuning notebook:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_dir = "outputs/finetune/qwen_0_5b_lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, adapter_dir)
```

## Intended Use

- Demonstrate PEFT fine-tuning for an educational GenAI tutor.
- Improve response structure for tutor, examiner, and critic personas.
- Work with RAG citations for factual answers about course lectures.

## Limitations

- This is an adapter, not a standalone model.
- It should not be used without the base Qwen model.
- It is not a substitute for RAG grounding; course facts should still come from retrieved lecture chunks.
- Safety rules in `src/agents/safety_agent.py` still govern cheating, plagiarism, and harmful requests.
### Framework versions

- PEFT 0.19.1