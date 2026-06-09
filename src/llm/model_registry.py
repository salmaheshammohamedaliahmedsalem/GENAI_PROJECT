from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path

from src.config import (
    CHAT_MODEL,
    FINETUNE_BASE_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LOCAL_MODEL_ALLOW_DOWNLOADS,
    OPENAI_API_KEY,
    OUTPUTS_DIR,
    ROOT_DIR,
)


@dataclass(frozen=True)
class ChatModelOption:
    id: str
    label: str
    kind: str
    available: bool
    status: str
    is_finetuned: bool = False
    path: str | None = None
    base_model: str | None = None

    def model_dump(self) -> dict:
        return asdict(self)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _base_model_dependency_status() -> tuple[bool, str]:
    required = ["torch", "transformers", "accelerate"]
    missing = [name for name in required if not _has_module(name)]
    if missing:
        return False, "Missing local model packages: " + ", ".join(missing)
    return True, "Ready for local base-model inference"


def _lora_dependency_status() -> tuple[bool, str]:
    base_ok, base_status = _base_model_dependency_status()
    if not base_ok:
        return base_ok, base_status
    if not _has_module("peft"):
        return False, "Missing local LoRA package: peft"
    return True, "Ready for local PEFT/LoRA inference"


def _cached_model_file_status(model_id: str, filenames: tuple[str, ...]) -> tuple[bool, str]:
    try:
        from transformers.utils import cached_file
    except Exception as exc:
        return False, f"Cannot inspect Hugging Face cache: {type(exc).__name__}: {exc}"

    for filename in filenames:
        try:
            cached_file(model_id, filename, local_files_only=True)
            return True, f"Cached {filename}"
        except Exception:
            continue
    return False, f"Missing cached files: {', '.join(filenames)}"


def _base_model_cache_status(base_model_id: str, require_tokenizer: bool = True) -> tuple[bool, str]:
    if LOCAL_MODEL_ALLOW_DOWNLOADS:
        return True, f"Base model {base_model_id} can be downloaded on first use"

    required_groups = [
        ("config", ("config.json",)),
        (
            "weights",
            (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            ),
        ),
    ]
    if require_tokenizer:
        required_groups.append(("tokenizer", ("tokenizer_config.json", "tokenizer.json")))

    missing = []
    for label, filenames in required_groups:
        ok, status = _cached_model_file_status(base_model_id, filenames)
        if not ok:
            missing.append(f"{label} ({status})")

    if missing:
        return (
            False,
            f"Base model {base_model_id} is not fully cached locally: " + "; ".join(missing),
        )
    return True, f"Base model {base_model_id} is cached locally"


def _local_base_model_status(base_model_id: str) -> tuple[bool, str]:
    deps_ok, deps_status = _base_model_dependency_status()
    if not deps_ok:
        return deps_ok, deps_status
    cache_ok, cache_status = _base_model_cache_status(base_model_id)
    return cache_ok, f"{deps_status}; {cache_status}"


def _local_lora_status(adapter_dir: Path, base_model_id: str) -> tuple[bool, str]:
    deps_ok, deps_status = _lora_dependency_status()
    if not deps_ok:
        return deps_ok, deps_status
    if not (adapter_dir / "adapter_model.safetensors").exists():
        return False, f"Missing LoRA adapter weights at {adapter_dir / 'adapter_model.safetensors'}"

    adapter_has_tokenizer = (adapter_dir / "tokenizer_config.json").exists() or (adapter_dir / "tokenizer.json").exists()
    cache_ok, cache_status = _base_model_cache_status(base_model_id, require_tokenizer=not adapter_has_tokenizer)
    return cache_ok, f"{deps_status}; {cache_status}; base model: {base_model_id}"


def _adapter_base_model(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        return FINETUNE_BASE_MODEL
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return data.get("base_model_name_or_path") or FINETUNE_BASE_MODEL
    except Exception:
        return FINETUNE_BASE_MODEL


def discover_finetuned_adapters() -> list[Path]:
    root = OUTPUTS_DIR / "finetune"
    if not root.exists():
        return []
    adapters = []
    for adapter_model in root.rglob("adapter_model.safetensors"):
        adapter_dir = adapter_model.parent
        if any(part.endswith("checkpoints") or part == "checkpoints" for part in adapter_dir.parts):
            continue
        if (adapter_dir / "adapter_config.json").exists():
            adapters.append(adapter_dir)

    def sort_key(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        priority = 0 if name == "qwen_0_5b_lora_adapter_salma" else 1
        return priority, str(path.relative_to(ROOT_DIR))

    return sorted(set(adapters), key=sort_key)


def list_chat_model_options(include_unavailable: bool = True) -> list[ChatModelOption]:
    options: list[ChatModelOption] = []
    base_ok, base_status = _local_base_model_status(FINETUNE_BASE_MODEL)

    for adapter_dir in discover_finetuned_adapters():
        rel_path = adapter_dir.relative_to(ROOT_DIR).as_posix()
        base_model = _adapter_base_model(adapter_dir)
        lora_ok, lora_status = _local_lora_status(adapter_dir, base_model)
        label_prefix = "Recommended fine-tuned model" if adapter_dir.name == "qwen_0_5b_lora_adapter_salma" else "Fine-tuned LoRA adapter"
        options.append(
            ChatModelOption(
                id=f"lora::{rel_path}",
                label=f"{label_prefix}: {adapter_dir.name}",
                kind="lora_adapter",
                available=lora_ok,
                status=lora_status,
                is_finetuned=True,
                path=str(adapter_dir),
                base_model=base_model,
            )
        )

    options.append(
        ChatModelOption(
            id=f"base::{FINETUNE_BASE_MODEL}",
            label=f"Base model only: {FINETUNE_BASE_MODEL}",
            kind="base_model",
            available=base_ok,
            status=f"{base_status}; no LoRA adapter applied",
            is_finetuned=False,
            base_model=FINETUNE_BASE_MODEL,
        )
    )

    if GROQ_API_KEY:
        options.append(
            ChatModelOption(
                id=f"groq::{GROQ_MODEL}",
                label=f"Groq hosted model: {GROQ_MODEL}",
                kind="groq",
                available=True,
                status="Uses GROQ_API_KEY through Groq's OpenAI-compatible endpoint",
                base_model=GROQ_MODEL,
            )
        )

    if OPENAI_API_KEY:
        options.append(
            ChatModelOption(
                id=f"openai::{CHAT_MODEL}",
                label=f"OpenAI hosted model: {CHAT_MODEL}",
                kind="openai",
                available=True,
                status="Uses OPENAI_API_KEY from the environment",
                base_model=CHAT_MODEL,
            )
        )

    options.append(
        ChatModelOption(
            id="local::rule_based",
            label="Local deterministic fallback",
            kind="local_rule_based",
            available=True,
            status="No neural model; useful only when APIs/local model packages are unavailable",
        )
    )

    if include_unavailable:
        return options
    return [option for option in options if option.available]


def get_recommended_chat_model_id() -> str:
    available = list_chat_model_options(include_unavailable=False)
    finetuned = [option for option in available if option.is_finetuned]
    if finetuned:
        return finetuned[0].id
    hosted = [option for option in available if option.kind in {"groq", "openai"}]
    if hosted:
        return hosted[0].id
    return "local::rule_based"


def runtime_default_chat_model_id() -> str:
    if GROQ_API_KEY:
        return f"groq::{GROQ_MODEL}"
    if OPENAI_API_KEY:
        return f"openai::{CHAT_MODEL}"
    return "local::rule_based"


def resolve_chat_model_option(model_id: str | None = None, prefer_finetuned: bool = False) -> ChatModelOption:
    target_id = model_id or (get_recommended_chat_model_id() if prefer_finetuned else runtime_default_chat_model_id())
    options = list_chat_model_options(include_unavailable=True)
    for option in options:
        if option.id == target_id:
            return option
    if target_id.startswith("lora::"):
        rel_path = target_id.removeprefix("lora::")
        adapter_dir = ROOT_DIR / rel_path
        base_model = _adapter_base_model(adapter_dir)
        lora_ok, lora_status = _local_lora_status(adapter_dir, base_model)
        return ChatModelOption(
            id=target_id,
            label=f"Fine-tuned LoRA adapter: {adapter_dir.name}",
            kind="lora_adapter",
            available=lora_ok,
            status=lora_status,
            is_finetuned=True,
            path=str(adapter_dir),
            base_model=base_model,
        )
    if target_id.startswith("base::"):
        base_model = target_id.removeprefix("base::") or FINETUNE_BASE_MODEL
        base_ok, base_status = _local_base_model_status(base_model)
        return ChatModelOption(
            id=target_id,
            label=f"Base model only: {base_model}",
            kind="base_model",
            available=base_ok,
            status=f"{base_status}; no LoRA adapter applied",
            base_model=base_model,
        )
    if target_id.startswith("groq::"):
        groq_model = target_id.removeprefix("groq::") or GROQ_MODEL
        return ChatModelOption(
            id=target_id,
            label=f"Groq hosted model: {groq_model}",
            kind="groq",
            available=bool(GROQ_API_KEY),
            status="Uses GROQ_API_KEY through Groq's OpenAI-compatible endpoint" if GROQ_API_KEY else "Missing GROQ_API_KEY",
            base_model=groq_model,
        )
    return next(option for option in options if option.id == "local::rule_based")
