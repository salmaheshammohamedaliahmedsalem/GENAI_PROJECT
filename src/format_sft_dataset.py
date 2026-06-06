import json
import os

def format_dataset():
    input_path = "data/finetuning/combined_dataset_clean.jsonl"
    output_path = "data/finetuning/sft_chat_dataset.jsonl"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    system_text = "You are an Adaptive GenAI Tutor. You can act as a tutor, examiner, or critic depending on the mode. Follow the requested structure exactly."
    
    count = 0
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {count+1}: {e}")
                continue
            
            mode = data.get("mode", "")
            instruction = data.get("instruction", "")
            user_input = data.get("input", "")
            output = data.get("output", "")
            
            user_text = f"Mode: {mode}\nInstruction: {instruction}\nInput:\n{user_input}"
            
            # Format in standard conversational messages structure
            formatted_item = {
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": output}
                ]
            }
            
            outfile.write(json.dumps(formatted_item, ensure_ascii=False) + "\n")
            count += 1
            
    print(f"Dataset formatted successfully! Converted {count} rows. Saved to: {output_path}")

if __name__ == "__main__":
    format_dataset()
