from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import FINETUNE_BASE_MODEL, OUTPUTS_DIR


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_with_lora(prompt: str, max_new_tokens: int = 256, adapter_dir: str | Path | None = None) -> str:
    adapter_path = Path(adapter_dir) if adapter_dir else OUTPUTS_DIR / "finetune" / "qwen_0_5b_lora_adapter"
    if not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"No LoRA adapter found at {adapter_path}")

    device = _device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").exists() else FINETUNE_BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(FINETUNE_BASE_MODEL, dtype=dtype)
    base_model.to(device)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    messages = [
        {"role": "system", "content": "You are an Adaptive GenAI Tutor. Follow the requested role and structure."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output_ids[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
