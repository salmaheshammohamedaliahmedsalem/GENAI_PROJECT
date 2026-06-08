from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path

from src.config import CHAT_MODEL, FINETUNE_BASE_MODEL, OPENAI_API_KEY, OUTPUTS_DIR, ROOT_DIR


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


def _local_model_dependency_status() -> tuple[bool, str]:
    required = ["torch", "transformers", "peft", "accelerate"]
    missing = [name for name in required if not _has_module(name)]
    if missing:
        return False, "Missing local model packages: " + ", ".join(missing)
    return True, "Ready for local PEFT/LoRA inference"


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
        if (adapter_dir / "adapter_config.json").exists():
            adapters.append(adapter_dir)

    def sort_key(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        priority = 0 if name == "qwen_0_5b_lora_adapter" else 1
        return priority, str(path.relative_to(ROOT_DIR))

    return sorted(set(adapters), key=sort_key)


def list_chat_model_options(include_unavailable: bool = True) -> list[ChatModelOption]:
    options: list[ChatModelOption] = []
    deps_ok, deps_status = _local_model_dependency_status()

    for adapter_dir in discover_finetuned_adapters():
        rel_path = adapter_dir.relative_to(ROOT_DIR).as_posix()
        base_model = _adapter_base_model(adapter_dir)
        label_prefix = "Recommended fine-tuned model" if adapter_dir.name == "qwen_0_5b_lora_adapter" else "Fine-tuned LoRA adapter"
        options.append(
            ChatModelOption(
                id=f"lora::{rel_path}",
                label=f"{label_prefix}: {adapter_dir.name}",
                kind="lora_adapter",
                available=deps_ok,
                status=f"{deps_status}; base model: {base_model}",
                is_finetuned=True,
                path=str(adapter_dir),
                base_model=base_model,
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
    hosted = [option for option in available if option.kind == "openai"]
    if hosted:
        return hosted[0].id
    return "local::rule_based"


def runtime_default_chat_model_id() -> str:
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
        deps_ok, deps_status = _local_model_dependency_status()
        return ChatModelOption(
            id=target_id,
            label=f"Fine-tuned LoRA adapter: {adapter_dir.name}",
            kind="lora_adapter",
            available=deps_ok and (adapter_dir / "adapter_model.safetensors").exists(),
            status=deps_status,
            is_finetuned=True,
            path=str(adapter_dir),
            base_model=_adapter_base_model(adapter_dir),
        )
    return next(option for option in options if option.id == "local::rule_based")
