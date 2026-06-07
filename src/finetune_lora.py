import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

# Configuration
MAX_TRAIN_EXAMPLES = 300  # Set to None for full training
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "outputs/finetune/qwen_0_5b_lora_adapter"

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )

def main():
    print(f"Initializing LoRA SFT pipeline for {MODEL_ID}...")
    
    # 1. Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Detected Mac Apple Silicon GPU. Using MPS.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Detected NVIDIA GPU. Using CUDA.")
    else:
        device = torch.device("cpu")
        print("No GPU detected. Using CPU.")

    # 2. Tokenizer Loading
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set padding side to right for SFT
    tokenizer.padding_side = "right"

    # 3. Model Loading with bitsandbytes fallback
    model = None
    
    # Attempt 4-bit bitsandbytes loading only on CUDA, since MPS doesn't support bitsandbytes natively
    attempt_4bit = (device.type == "cuda")
    
    if attempt_4bit:
        try:
            print("Attempting to load model in 4-bit quantization with bitsandbytes...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            print("Model loaded successfully in 4-bit quantization!")
        except Exception as e:
            print(f"4-bit quantization failed: {e}. Falling back to normal model loading...")
            model = None

    if model is None:
        # Load in half-precision for GPU, or full precision for CPU
        dtype = torch.float16 if device.type in ["mps", "cuda"] else torch.float32
        print(f"Loading model with standard precision: {dtype}...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map="auto" if device.type != "mps" else None,
                trust_remote_code=True,
            )
            if device.type == "mps":
                print("Moving model weights to MPS device manually...")
                model = model.to(device)
            print(f"Model loaded successfully in standard precision on {device}!")
        except Exception as e:
            print(f"Failed to load model with {dtype}: {e}. Trying float32 loading...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            if device.type in ["mps", "cuda"]:
                model = model.to(device)
            print("Model loaded in float32 successfully.")

    # 4. Dataset Loading & Preprocessing
    dataset_path = "data/finetuning/sft_chat_dataset.jsonl"
    print(f"Loading dataset from: {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    if MAX_TRAIN_EXAMPLES is not None:
        print(f"Smoke test mode active. Limiting dataset to first {MAX_TRAIN_EXAMPLES} examples.")
        dataset = dataset.select(range(min(MAX_TRAIN_EXAMPLES, len(dataset))))
    
    print(f"Total dataset size for SFT: {len(dataset)}")
    
    # Pre-apply chat template using the Qwen tokenizer to generate formatted strings
    def apply_template(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}
    
    print("Formatting SFT messages using tokenizer chat template...")
    dataset = dataset.map(apply_template, remove_columns=["messages"])
    
    # 5. Configure LoRA
    print("Configuring LoRA adapter...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Enable gradient checkpointing to save VRAM/Memory if standard loading was used
    if hasattr(model, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    # 6. Training Arguments
    # Batch size 1 is highly recommended for MPS/Mac memory footprint
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 8
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        num_train_epochs=1,
        optim="adamw_torch",
        gradient_checkpointing=True,
        fp16=(device.type == "cuda"), # Use fp16 for CUDA, not for MPS where it can be unstable
        remove_unused_columns=False,
        report_to="none", # Disable logging to wandb/tensorboard to avoid extra dependencies
    )

    # 7. SFT Trainer Initialization
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 8. Start Fine-Tuning
    print("Starting SFT training...")
    trainer.train()
    
    # 9. Save final adapter
    print(f"Training complete. Saving adapter to: {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fine-tuning pipeline completed successfully!")

if __name__ == "__main__":
    main()
