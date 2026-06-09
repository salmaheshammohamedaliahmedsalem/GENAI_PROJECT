from pathlib import Path
from functools import lru_cache

from src.config import FINETUNE_BASE_MODEL, LOCAL_MODEL_ALLOW_DOWNLOADS, OUTPUTS_DIR


def _load_base_dependencies():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


def _local_files_only(local_files_only: bool | None) -> bool:
    if local_files_only is None:
        return not LOCAL_MODEL_ALLOW_DOWNLOADS
    return local_files_only


def _load_lora_dependencies():
    from peft import PeftModel

    torch, AutoModelForCausalLM, AutoTokenizer = _load_base_dependencies()
    return torch, PeftModel, AutoModelForCausalLM, AutoTokenizer


def _device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_base_model(AutoModelForCausalLM, base_model_id: str, dtype, local_files_only: bool):
    try:
        return AutoModelForCausalLM.from_pretrained(base_model_id, dtype=dtype, local_files_only=local_files_only)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype, local_files_only=local_files_only)


@lru_cache(maxsize=1)
def _load_base_components(base_model_id: str, local_files_only: bool):
    torch, AutoModelForCausalLM, AutoTokenizer = _load_base_dependencies()
    device = _device(torch)
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_base_model(AutoModelForCausalLM, base_model_id, dtype, local_files_only)
    model.to(device)
    model.eval()
    return tokenizer, model, device, torch


@lru_cache(maxsize=1)
def _load_lora_components(adapter_dir: str, base_model_id: str, local_files_only: bool):
    torch, PeftModel, AutoModelForCausalLM, AutoTokenizer = _load_lora_dependencies()
    adapter_path = Path(adapter_dir)
    if not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"No LoRA adapter found at {adapter_path}")

    device = _device(torch)
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").exists() else base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = _load_base_model(AutoModelForCausalLM, base_model_id, dtype, local_files_only)
    base_model.to(device)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model, device, torch


def _messages_to_text(tokenizer, messages: list[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = "\n".join(f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages)
    return text + "\nassistant:"


def _generate_from_components(tokenizer, model, device, torch, messages: list[dict], max_new_tokens: int) -> str:
    text = _messages_to_text(tokenizer, messages)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_with_base_messages(
    messages: list[dict],
    max_new_tokens: int = 384,
    base_model_id: str = FINETUNE_BASE_MODEL,
    local_files_only: bool | None = None,
) -> str:
    local_files_only = _local_files_only(local_files_only)
    tokenizer, model, device, torch = _load_base_components(base_model_id, local_files_only)
    return _generate_from_components(tokenizer, model, device, torch, messages, max_new_tokens)


def generate_with_lora_messages(
    messages: list[dict],
    max_new_tokens: int = 384,
    adapter_dir: str | Path | None = None,
    base_model_id: str = FINETUNE_BASE_MODEL,
    local_files_only: bool | None = None,
) -> str:
    local_files_only = _local_files_only(local_files_only)
    adapter_path = Path(adapter_dir) if adapter_dir else OUTPUTS_DIR / "finetune" / "qwen_0_5b_lora_adapter_salma"
    tokenizer, model, device, torch = _load_lora_components(str(adapter_path), base_model_id, local_files_only)
    return _generate_from_components(tokenizer, model, device, torch, messages, max_new_tokens)


def generate_with_lora(prompt: str, max_new_tokens: int = 256, adapter_dir: str | Path | None = None) -> str:
    messages = [
        {"role": "system", "content": "You are an Adaptive GenAI Tutor. Follow the requested role and structure."},
        {"role": "user", "content": prompt},
    ]
    return generate_with_lora_messages(messages, max_new_tokens=max_new_tokens, adapter_dir=adapter_dir)
