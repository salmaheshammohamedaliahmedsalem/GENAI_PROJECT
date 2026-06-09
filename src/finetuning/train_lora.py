import json
import os
from src.config import FINETUNE_BASE_MODEL, FINETUNE_DIR, OUTPUTS_DIR

def _device_status(torch):
    cuda_available = torch.cuda.is_available()
    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if cuda_available:
        return "cuda", {
            "cuda_available": True,
            "mps_built": mps_built,
            "mps_available": mps_available,
        }
    if mps_available:
        return "mps", {
            "cuda_available": False,
            "mps_built": mps_built,
            "mps_available": True,
        }
    return None, {
        "cuda_available": False,
        "mps_built": mps_built,
        "mps_available": mps_available,
    }

def train_lora() -> None:
    log_dir = OUTPUTS_DIR / "finetune"
    log_dir.mkdir(parents=True, exist_ok=True)
    max_train_examples = int(os.getenv("FINETUNE_MAX_TRAIN_EXAMPLES", "0"))
    max_eval_examples = int(os.getenv("FINETUNE_MAX_EVAL_EXAMPLES", "0"))
    max_length = int(os.getenv("FINETUNE_MAX_LENGTH", "1024"))
    epochs = float(os.getenv("FINETUNE_EPOCHS", "1"))
    try:
        import torch
        device, device_info = _device_status(torch)
        if device is None:
            print("No CUDA or MPS device detected. LoRA training skipped gracefully.")
            print("Dataset is still ready in data/finetune/. Use an MPS-enabled Python runtime or Colab/GPU for training.")
            (log_dir / "training_log.json").write_text(json.dumps({
                "status": "skipped",
                "reason": "No CUDA or MPS device detected",
                "device": None,
                "device_info": device_info,
                "dataset_dir": str(FINETUNE_DIR),
                "base_model": FINETUNE_BASE_MODEL,
            }, indent=2), encoding="utf-8")
            return
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
        from peft import LoraConfig, get_peft_model

        dataset = load_dataset("json", data_files={
            "train": str(FINETUNE_DIR / "train.jsonl"),
            "validation": str(FINETUNE_DIR / "val.jsonl"),
        })
        if max_train_examples > 0:
            dataset["train"] = dataset["train"].select(range(min(max_train_examples, len(dataset["train"]))))
        if max_eval_examples > 0:
            dataset["validation"] = dataset["validation"].select(range(min(max_eval_examples, len(dataset["validation"]))))
        tokenizer = AutoTokenizer.from_pretrained(FINETUNE_BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(FINETUNE_BASE_MODEL, dtype=dtype)
        model.to(device)
        config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM")
        model = get_peft_model(model, config)

        def format_row(row):
            if row.get("messages"):
                text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
            else:
                text = f"Instruction: {row['instruction']}\nContext: {row['context']}\nResponse: {row['response']}"
            return tokenizer(text, truncation=True, max_length=max_length)

        tokenized = dataset.map(format_row, remove_columns=dataset["train"].column_names)
        args = TrainingArguments(
            output_dir=str(OUTPUTS_DIR / "finetune" / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=50,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()
        adapter_name = os.getenv("FINETUNE_OUTPUT_ADAPTER_DIR", "lora_adapter_salma")
        adapter_dir = log_dir / adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        (log_dir / "training_log.json").write_text(json.dumps({
            "status": "completed",
            "device": device,
            "device_info": device_info,
            "adapter_dir": str(adapter_dir),
            "base_model": FINETUNE_BASE_MODEL,
            "train_examples": len(dataset["train"]),
            "validation_examples": len(dataset["validation"]),
            "epochs": epochs,
            "max_length": max_length,
        }, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"LoRA training skipped due to environment issue: {e}")
        (log_dir / "training_log.json").write_text(json.dumps({
            "status": "skipped",
            "reason": str(e),
            "dataset_dir": str(FINETUNE_DIR),
            "base_model": FINETUNE_BASE_MODEL,
        }, indent=2), encoding="utf-8")
