import json
from pathlib import Path

# Import paths and utilities
from utils import (
    FINETUNING_DIR,
    get_logger
)

logger = get_logger("validate_dataset")

def validate_tutor_output(output: str) -> bool:
    """
    Validates that a Tutor mode output contains the five required sections.
    """
    required_sections = [
        "Simple explanation:",
        "Analogy:",
        "Course-grounded answer:",
        "Common misconception:",
        "Quick check question:"
    ]
    return all(section in output for section in required_sections)

def validate_examiner_output(output: str) -> bool:
    """
    Validates that an Examiner mode output contains either:
    1. Question generation sections (Question, Choices/Explanation/Guidance)
    2. Answer grading sections (Score, Feedback, Corrected answer)
    """
    is_question_gen = "Question:" in output and ("Choices:" in output or "Correct answer:" in output or "Answer Guidance:" in output)
    is_grading = "Score:" in output and "Feedback:" in output
    return is_question_gen or is_grading

def validate_critic_output(output: str) -> bool:
    """
    Validates that a Critic mode output contains the required critique and improved answer.
    """
    return "Critique:" in output and "Improved answer:" in output

def validate_example(item: dict) -> tuple:
    """
    Validates a single training example dictionary.
    Returns (is_valid, reason).
    """
    # 1. Check for basic fields
    instruction = item.get("instruction", "").strip()
    input_str = item.get("input", "").strip()
    output_str = item.get("output", "").strip()
    mode = item.get("mode", "").strip()
    
    if not instruction:
        return False, "Missing or empty instruction"
    if not input_str:
        return False, "Missing or empty input"
    if not output_str:
        return False, "Missing or empty output"
    if not mode:
        return False, "Missing or empty mode"
        
    # 2. Check mode specific format constraints
    if mode == "tutor":
        if not validate_tutor_output(output_str):
            return False, "Tutor output missing one or more of the 5 required sections"
    elif mode == "examiner":
        if not validate_examiner_output(output_str):
            return False, "Examiner output missing either Question or Score/Feedback structure"
    elif mode == "critic":
        if not validate_critic_output(output_str):
            return False, "Critic output missing Critique or Improved answer structure"
    else:
        return False, f"Unknown mode: {mode}"
        
    return True, "Valid"

def process_and_clean_file(raw_file: Path, clean_file: Path) -> dict:
    """
    Validates, deduplicates, and saves clean records for a JSONL dataset.
    Returns statistical metrics.
    """
    if not raw_file.exists():
        logger.warning(f"File {raw_file} does not exist. Skipping.")
        return {}
        
    records = []
    invalid_count = 0
    invalid_reasons = {}
    
    # Read raw JSONL file
    with open(raw_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip())
                is_valid, reason = validate_example(item)
                if is_valid:
                    records.append(item)
                else:
                    invalid_count += 1
                    invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            except Exception as e:
                invalid_count += 1
                reason = f"JSON decode error: {e}"
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                
    total_raw = len(records) + invalid_count
    
    # Deduplicate based on unique (instruction, input) hash
    seen_keys = set()
    dedup_records = []
    dup_count = 0
    
    for r in records:
        unique_key = (r["instruction"], r["input"])
        if unique_key not in seen_keys:
            seen_keys.add(unique_key)
            dedup_records.append(r)
        else:
            dup_count += 1
            
    # Compute averages
    avg_input_len = 0
    avg_output_len = 0
    if dedup_records:
        avg_input_len = sum(len(r["input"]) for r in dedup_records) / len(dedup_records)
        avg_output_len = sum(len(r["output"]) for r in dedup_records) / len(dedup_records)
        
    # Save cleaned file
    try:
        with open(clean_file, "w", encoding="utf-8") as f:
            for r in dedup_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Saved cleaned file with {len(dedup_records)} rows to {clean_file}")
    except Exception as e:
        logger.error(f"Failed to save cleaned file {clean_file}: {e}")
        
    return {
        "total_raw": total_raw,
        "valid_count": len(records),
        "dedup_count": len(dedup_records),
        "invalid_count": invalid_count,
        "duplicate_count": dup_count,
        "avg_input_len": avg_input_len,
        "avg_output_len": avg_output_len,
        "invalid_reasons": invalid_reasons
    }

def main():
    logger.info("Starting dataset validation and cleaning pipeline...")
    
    datasets = {
        "tutor": ("tutor_dataset.jsonl", "tutor_dataset_clean.jsonl"),
        "examiner": ("examiner_dataset.jsonl", "examiner_dataset_clean.jsonl"),
        "critic": ("critic_dataset.jsonl", "critic_dataset_clean.jsonl"),
        "combined": ("combined_dataset.jsonl", "combined_dataset_clean.jsonl")
    }
    
    stats_summary = {}
    
    for mode_name, (raw_name, clean_name) in datasets.items():
        logger.info(f"Validating dataset: {mode_name}...")
        raw_path = FINETUNING_DIR / raw_name
        clean_path = FINETUNING_DIR / clean_name
        
        stats = process_and_clean_file(raw_path, clean_path)
        if stats:
            stats_summary[mode_name] = stats
            
    # Print out beautiful dataset statistics summary
    print("\n" + "="*60)
    print("           FINE-TUNING DATASET VALIDATION SUMMARY")
    print("="*60)
    
    for mode, s in stats_summary.items():
        print(f"\nDataset: {mode.upper()}")
        print(f"  • Total Loaded Rows: {s['total_raw']}")
        print(f"  • Valid Rows:        {s['valid_count']}")
        print(f"  • Invalid Rows:      {s['invalid_count']} (Filtered)")
        print(f"  • Duplicate Rows:    {s['duplicate_count']} (Filtered)")
        print(f"  • Cleaned Rows Saved:{s['dedup_count']}")
        print(f"  • Avg Input Length:  {s['avg_input_len']:.1f} characters")
        print(f"  • Avg Output Length: {s['avg_output_len']:.1f} characters")
        
        if s['invalid_reasons']:
            print("  • Invalid Reasons:")
            for reason, count in s['invalid_reasons'].items():
                print(f"    - {reason}: {count} occurrences")
                
    print("="*60 + "\n")
    logger.info("Dataset validation and cleaning completed successfully.")

if __name__ == "__main__":
    main()
