import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

def main():
    print("Initializing inference environment for LoRA testing...")
    
    base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_dir = "outputs/finetune/qwen_0_5b_lora_adapter_salma"
    output_file = "results/lora_test_outputs.md"
    
    # 1. Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # 2. Tokenizer Loading
    print(f"Loading tokenizer from {lora_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Model Loading
    dtype = torch.float16 if device.type in ["mps", "cuda"] else torch.float32
    print(f"Loading base model: {base_model_id} in {dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if device.type != "mps" else None,
        trust_remote_code=True,
    )
    if device.type == "mps":
        base_model = base_model.to(device)

    print(f"Loading PEFT adapter from {lora_dir}...")
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()
    print("Model loaded and set to evaluation mode successfully.")

    # 4. 10 Curated Prompts
    system_text = "You are an Adaptive GenAI Tutor. You can act as a tutor, examiner, or critic depending on the mode. Follow the requested structure exactly."
    
    test_prompts = [
        # Tutor Mode
        {
            "id": 1,
            "mode": "tutor",
            "instruction": "Teach the concept using the required tutoring format.",
            "input": "Student level: beginner\nQuestion: What is an LLM context window and why does it matter?",
            "title": "Tutor Mode (Beginner) - Context Window"
        },
        {
            "id": 2,
            "mode": "tutor",
            "instruction": "Teach the concept using the required tutoring format.",
            "input": "Student level: intermediate\nQuestion: How does self-attention work in Transformer models?",
            "title": "Tutor Mode (Intermediate) - Self-Attention"
        },
        {
            "id": 3,
            "mode": "tutor",
            "instruction": "Teach the concept using the required tutoring format.",
            "input": "Student level: advanced\nQuestion: What is decoding temperature and how does it affect creativity?",
            "title": "Tutor Mode (Advanced) - Decoding Temperature"
        },
        {
            "id": 4,
            "mode": "tutor",
            "instruction": "Teach the concept using the required tutoring format.",
            "input": "Student level: intermediate\nQuestion: Explain prompt injections and how to mitigate them.",
            "title": "Tutor Mode (Intermediate) - Prompt Injection"
        },
        
        # Examiner Mode
        {
            "id": 5,
            "mode": "examiner",
            "instruction": "Evaluate the student's answer and explain any gaps.",
            "input": "Student level: beginner\nQuestion: How does gradient descent work?\nStudent Answer: Gradient descent works by choosing a completely random direction and walking down the hill until you find the bottom.",
            "title": "Examiner Mode (Beginner) - Gradient Descent"
        },
        {
            "id": 6,
            "mode": "examiner",
            "instruction": "Evaluate the student's answer and explain any gaps.",
            "input": "Student level: intermediate\nQuestion: What is an epoch in deep learning?\nStudent Answer: An epoch is the number of batches we train our model on before we update the weights.",
            "title": "Examiner Mode (Intermediate) - Epoch Definition"
        },
        {
            "id": 7,
            "mode": "examiner",
            "instruction": "Test the student's knowledge and ask a follow-up question.",
            "input": "Student level: advanced\nQuestion: What is the main difference between SFT and RLHF in alignment?\nStudent Answer: SFT uses human demonstration examples to train the model to mimic positive behaviors, while RLHF uses reinforcement learning with a reward model to optimize the policy based on human preferences.",
            "title": "Examiner Mode (Advanced) - SFT vs RLHF"
        },

        # Critic Mode
        {
            "id": 8,
            "mode": "critic",
            "instruction": "Critique the provided code and suggest improvements.",
            "input": "Code language: Python\nCode snippet:\n```python\ndef matrix_multiply(A, B):\n    result = []\n    for i in range(len(A)):\n        row = []\n        for j in range(len(B[0])):\n            val = 0\n            for k in range(len(B)):\n                val += A[i][k] * B[k][j]\n            row.append(val)\n        result.append(row)\n    return result\n```\nTask: Make a critique and suggest optimizations.",
            "title": "Critic Mode - Matrix Multiplication Code"
        },
        {
            "id": 9,
            "mode": "critic",
            "instruction": "Critique the student's statement for misconceptions.",
            "input": "Statement: Bigger models are always better than smaller ones, and if a model has over 100B parameters, it will never hallucinate or make factual errors because it has stored all facts in its weights.",
            "title": "Critic Mode - Model Scaling & Hallucinations"
        },
        {
            "id": 10,
            "mode": "critic",
            "instruction": "Critique the project design and provide suggestions.",
            "input": "Project idea: I want to build a real-time medical diagnostic chatbot by fine-tuning an 8B model on 50 pages of web articles I scraped. It needs to give 100% correct medical diagnoses to patients.",
            "title": "Critic Mode - Medical Chatbot Project Design"
        }
    ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write Header
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# LoRA Fine-Tuned Model Inference Outputs\n\n")
        f.write("This file contains the outputs generated by the fine-tuned `Qwen2.5-0.5B-Instruct` model (using LoRA) on 10 hand-curated prompts across three modes: Tutor, Examiner, and Critic.\n\n")
        f.write("## Inference Settings\n")
        f.write("- **Model ID**: `Qwen/Qwen2.5-0.5B-Instruct`\n")
        f.write("- **Adapter Path**: `outputs/finetune/qwen_0_5b_lora_adapter_salma`\n")
        f.write("- **Parameters**: temperature = 0.7, top_p = 0.9, max_new_tokens = 512\n\n")
        f.write("---\n\n")

    print("Running inference on test prompts...")
    for idx, prompt in enumerate(tqdm(test_prompts)):
        user_content = f"Mode: {prompt['mode']}\nInstruction: {prompt['instruction']}\nInput:\n{prompt['input']}"
        
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content}
        ]
        
        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        # Extract assistant response only
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        
        # Append to output markdown file
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"## Test {prompt['id']}: {prompt['title']}\n\n")
            f.write(f"**Mode**: `{prompt['mode']}`  \n")
            f.write(f"**Instruction**: {prompt['instruction']}  \n\n")
            f.write("**User Input**:\n")
            f.write("```\n")
            f.write(f"{prompt['input']}\n")
            f.write("```\n\n")
            f.write("**LoRA Tutor Model Output**:\n")
            f.write(f"{response}\n\n")
            f.write("---\n\n")
            
    print(f"LoRA inference testing complete! Outputs saved to: {output_file}")

if __name__ == "__main__":
    main()
