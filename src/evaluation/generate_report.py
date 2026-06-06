import pandas as pd
from src.config import OUTPUTS_DIR
from src.evaluation.run_baselines import run_baselines
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.evaluation.evaluate_answers import evaluate_answers
from src.evaluation.evaluate_safety import evaluate_safety

def _metrics_table(metrics: dict) -> str:
    rows = ["| Metric | Value |", "| --- | ---: |"]
    for key, value in metrics.items():
        label = key.replace("_", " ").title()
        rows.append(f"| {label} | {value} |")
    return "\n".join(rows)

def _answer_table(rows: list[dict]) -> str:
    table = ["| Question | Correctness | Groundedness | Clarity | Usefulness |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        table.append(
            f"| {row['question']} | {row['correctness']} | {row['groundedness']} | {row['clarity']} | {row['usefulness']} |"
        )
    return "\n".join(table)

def _safety_table(rows: list[dict]) -> str:
    table = ["| Test | Refused |", "| --- | --- |"]
    for row in rows:
        table.append(f"| {row['test']} | {'Yes' if row['refused'] else 'No'} |")
    return "\n".join(table)

def generate_report() -> None:
    out = OUTPUTS_DIR / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    rows = run_baselines()
    retrieval = evaluate_retrieval(rows)
    answers = evaluate_answers(rows)
    safety = evaluate_safety()
    df = pd.DataFrame(rows)
    df.to_csv(out / "evaluation_results.csv", index=False)
    md = f"""# Evaluation Summary

## Baselines
This demo evaluation runs the final system on representative questions.

## Retrieval Metrics
{_metrics_table(retrieval)}

## Answer Evaluation
{_answer_table(answers["rows"])}

## Safety Evaluation
Safety pass rate: **{safety["pass_rate"]:.0%}**

{_safety_table(safety["safety_results"])}

## Strengths
- Clear routing
- Grounded answers when sources exist
- Tool use for calculation
- Safety refusals

## Weaknesses
- Online RAG depends on available search package/API
- LoRA training requires GPU
- Automatic evaluation is simplified
"""
    (out / "evaluation_summary.md").write_text(md, encoding="utf-8")
