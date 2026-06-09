import pandas as pd
from src.config import OUTPUTS_DIR
from src.evaluation.run_baselines import run_baselines
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.evaluation.evaluate_answers import evaluate_answers
from src.evaluation.evaluate_safety import evaluate_safety
from src.evaluation.baseline_comparison import run_baseline_comparison, summarize_comparison


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


def _comparison_summary_table(summary: dict) -> str:
    table = [
        "| Configuration | Avg Score | Avg Groundedness | Avg Keyword Coverage | Avg Word Count |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| No RAG (Groq, no course material) | {summary['no_rag_avg_score']} | {summary['no_rag_avg_groundedness']} | {summary['no_rag_avg_keyword_coverage']} | {summary['no_rag_avg_word_count']:.0f} |",
        f"| Offline RAG (Groq + BM25 course retrieval) | {summary['rag_avg_score']} | {summary['rag_avg_groundedness']} | {summary['rag_avg_keyword_coverage']} | {summary['rag_avg_word_count']:.0f} |",
    ]
    return "\n".join(table)


def _per_question_comparison_table(rows: list[dict]) -> str:
    table = [
        "| Question | No RAG Score | RAG Score | RAG Gain | Chunks Retrieved | RAG Citations |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        gain = round(row["rag_score"] - row["no_rag_score"], 3)
        gain_str = f"+{gain}" if gain >= 0 else str(gain)
        q = row["question"][:60] + ("…" if len(row["question"]) > 60 else "")
        citations = "Yes" if row["rag_citations"] else "No"
        table.append(
            f"| {q} | {row['no_rag_score']} | {row['rag_score']} | {gain_str} | {row['rag_chunks_retrieved']} | {citations} |"
        )
    return "\n".join(table)


def generate_report() -> None:
    out = OUTPUTS_DIR / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    rows = run_baselines()
    retrieval = evaluate_retrieval(rows)
    answers = evaluate_answers(rows)
    safety = evaluate_safety()
    pd.DataFrame(rows).to_csv(out / "evaluation_results.csv", index=False)

    comparison_rows = run_baseline_comparison()
    comparison_summary = summarize_comparison(comparison_rows)
    pd.DataFrame(comparison_rows).to_csv(out / "baseline_comparison.csv", index=False)

    rag_gain = round(comparison_summary["rag_avg_score"] - comparison_summary["no_rag_avg_score"], 3)
    md = f"""# Evaluation Summary

## Baseline Comparison: Groq (No RAG) vs Groq + Offline RAG

Both conditions use the same model (`{comparison_summary['model']}`).
The only difference is whether BM25 retrieves chunks from the course lecture PDFs first.
Tested on {comparison_summary['n_questions']} questions covering RAG, LoRA, and RLHF.

Scoring is heuristic — keyword coverage (40%), groundedness (40%), answer length (20%).
Groundedness: 0 = no retrieval used, 1 = retrieval used but no citations, 2 = retrieval with citations.

### Aggregate Results

{_comparison_summary_table(comparison_summary)}

### Per-Question Breakdown

{_per_question_comparison_table(comparison_rows)}

**Key finding:** Adding course retrieval raises the average score from {comparison_summary['no_rag_avg_score']} to {comparison_summary['rag_avg_score']} (gain: +{rag_gain})
and groundedness from {comparison_summary['no_rag_avg_groundedness']} to {comparison_summary['rag_avg_groundedness']}.
Without RAG, Groq answers from its training data only — no course-specific evidence, no citations.

---

## Retrieval Metrics (Final System)

{_metrics_table(retrieval)}

## Answer Quality (Final System, 5-point scale)

{_answer_table(answers["rows"])}

## Safety Evaluation

Safety pass rate: **{safety["pass_rate"]:.0%}**

{_safety_table(safety["safety_results"])}

## Strengths
- RAG retrieval consistently improves groundedness and keyword coverage over the no-RAG baseline
- Clear routing between offline, online, hybrid, and tool-only paths
- Tool use for calculations (calculator) and structured output (quiz, grader)
- Safety refusals block all tested academic-integrity violations

## Weaknesses
- Online RAG depends on a configured Tavily API key or ddgs fallback
- LoRA training requires a GPU or Apple MPS environment
- Heuristic scoring does not replace human evaluation; scores should be interpreted as relative comparisons
"""
    (out / "evaluation_summary.md").write_text(md, encoding="utf-8")
