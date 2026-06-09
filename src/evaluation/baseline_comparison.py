"""
Compare two configurations on the same test questions using Groq:
  no_rag      — Groq answers from parametric memory only, no course material retrieved
  offline_rag — BM25 retrieves chunks from lecture PDFs, Groq answers using that evidence
"""
import re
import time

from src.agents.adaptation_agent import StudentAdaptationAgent
from src.agents.tutor_agent import TutorAgent
from src.llm.model_registry import runtime_default_chat_model_id
from src.rag.hybrid_retriever import HybridRetriever

TEST_QUESTIONS = [
    {
        "question": "What is RAG and why does it reduce hallucinations?",
        "keywords": ["rag", "retrieval", "hallucination", "grounded", "context", "generation", "evidence"],
    },
    {
        "question": "Explain the difference between LoRA and full fine-tuning.",
        "keywords": ["lora", "fine-tuning", "parameters", "adapter", "rank", "weights", "peft"],
    },
    {
        "question": "How does RLHF align a language model with human preferences?",
        "keywords": ["rlhf", "reward", "human feedback", "policy", "alignment", "preference", "ppo"],
    },
]

_CALL_DELAY = 2.0  # seconds between Groq API calls to avoid rate limiting


def _score_answer(answer: str, keywords: list[str], retrieved_chunks: list) -> dict:
    answer_lower = answer.lower()
    has_citations = bool(re.findall(r"\[Source:", answer))
    word_count = len(answer.split())
    keyword_hits = sum(1 for kw in keywords if kw in answer_lower)
    keyword_coverage = round(keyword_hits / len(keywords), 2) if keywords else 0.0

    # 0 = no retrieval, 1 = retrieval without citations, 2 = retrieval with citations
    if not retrieved_chunks:
        groundedness = 0
    elif has_citations:
        groundedness = 2
    else:
        groundedness = 1

    is_error = any(p in answer_lower for p in ["rate limit", "could not be loaded", "error code"])
    length_bonus = 0.0 if is_error else min(1.0, max(0.0, (word_count - 20) / 150))

    # keyword coverage 40%, groundedness 40%, length 20%
    overall = round(keyword_coverage * 0.4 + (groundedness / 2) * 0.4 + length_bonus * 0.2, 3)

    return {
        "has_citations": has_citations,
        "word_count": word_count,
        "keyword_hits": keyword_hits,
        "keyword_coverage": keyword_coverage,
        "groundedness": groundedness,
        "overall_score": overall,
        "is_error": is_error,
    }


def _interpret(no_rag: dict, rag: dict, chunks: int, gain: float) -> str:
    parts = []

    # Groundedness change
    if no_rag["groundedness"] == 0 and rag["groundedness"] == 2:
        parts.append(
            f"Groundedness improved from 0 to 2: RAG retrieved {chunks} course chunks and the answer includes citations. "
            "Without RAG the model answered entirely from pretraining with no course-specific evidence."
        )
    elif no_rag["groundedness"] == 0 and rag["groundedness"] == 1:
        parts.append(
            f"Groundedness improved from 0 to 1: RAG retrieved {chunks} chunks but citations were not added to the answer. "
            "The model used retrieved context but did not explicitly cite the sources."
        )
    else:
        parts.append(f"Groundedness unchanged at {rag['groundedness']}.")

    # Keyword coverage
    kw_diff = round(rag["keyword_coverage"] - no_rag["keyword_coverage"], 2)
    if kw_diff >= 0.1:
        parts.append(
            f"Keyword coverage increased by {kw_diff}: RAG context introduced more course-specific terminology into the answer."
        )
    elif kw_diff <= -0.1:
        parts.append(
            f"Keyword coverage dropped by {abs(kw_diff)}: the no-RAG answer used more of the expected terms from pretraining, "
            "but lacked citations and course grounding."
        )
    else:
        parts.append("Keyword coverage was similar in both conditions.")

    # Overall gain
    if gain >= 0.3:
        parts.append(f"Overall score gain of +{gain}: RAG substantially improved this answer.")
    elif gain >= 0.1:
        parts.append(f"Overall score gain of +{gain}: RAG moderately improved this answer.")
    elif gain > 0:
        parts.append(f"Overall score gain of +{gain}: small improvement from RAG.")
    else:
        parts.append(f"Overall score change of {gain}: RAG did not improve the score on this question.")

    return " | ".join(parts)


def run_baseline_comparison() -> list[dict]:
    """Run no-RAG vs offline-RAG comparison using Groq for both conditions.
    A short delay is added between calls to stay within Groq's rate limit."""
    model_id = runtime_default_chat_model_id()
    tutor = TutorAgent()
    retriever = HybridRetriever()
    profile = StudentAdaptationAgent().run("intermediate")
    rows = []

    for item in TEST_QUESTIONS:
        question = item["question"]
        keywords = item["keywords"]

        # ── No RAG: Groq answers from parametric memory only ─────────────────
        no_rag_answer = tutor.answer(
            question, [], "no_retrieval",
            student_profile=profile,
            model_selection=model_id,
        )
        no_rag_scores = _score_answer(no_rag_answer, keywords, [])
        time.sleep(_CALL_DELAY)

        # ── Offline RAG: BM25 retrieval → Groq answers with evidence ─────────
        try:
            chunks = retriever.retrieve(question, mode="offline_only")
        except Exception:
            chunks = []
        rag_answer = tutor.answer(
            question, chunks, "offline_only",
            student_profile=profile,
            model_selection=model_id,
        )
        rag_scores = _score_answer(rag_answer, keywords, chunks)
        time.sleep(_CALL_DELAY)

        gain = round(rag_scores["overall_score"] - no_rag_scores["overall_score"], 3)
        interpretation = _interpret(no_rag_scores, rag_scores, len(chunks), gain)

        rows.append({
            "question": question,
            "model": model_id,
            # no rag
            "no_rag_answer_preview": no_rag_answer[:300].replace("\n", " "),
            "no_rag_groundedness": no_rag_scores["groundedness"],
            "no_rag_keyword_coverage": no_rag_scores["keyword_coverage"],
            "no_rag_citations": no_rag_scores["has_citations"],
            "no_rag_word_count": no_rag_scores["word_count"],
            "no_rag_score": no_rag_scores["overall_score"],
            "no_rag_error": no_rag_scores["is_error"],
            # offline rag
            "rag_chunks_retrieved": len(chunks),
            "rag_answer_preview": rag_answer[:300].replace("\n", " "),
            "rag_groundedness": rag_scores["groundedness"],
            "rag_keyword_coverage": rag_scores["keyword_coverage"],
            "rag_citations": rag_scores["has_citations"],
            "rag_word_count": rag_scores["word_count"],
            "rag_score": rag_scores["overall_score"],
            "rag_error": rag_scores["is_error"],
            # interpretation
            "rag_gain": gain,
            "interpretation": interpretation,
        })

    return rows


def summarize_comparison(rows: list[dict]) -> dict:
    def avg(key: str) -> float:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(values) / len(values), 3) if values else 0.0

    return {
        "model": rows[0]["model"] if rows else "unknown",
        "n_questions": len(rows),
        "no_rag_avg_score": avg("no_rag_score"),
        "no_rag_avg_groundedness": avg("no_rag_groundedness"),
        "no_rag_avg_keyword_coverage": avg("no_rag_keyword_coverage"),
        "no_rag_avg_word_count": avg("no_rag_word_count"),
        "rag_avg_score": avg("rag_score"),
        "rag_avg_groundedness": avg("rag_groundedness"),
        "rag_avg_keyword_coverage": avg("rag_keyword_coverage"),
        "rag_avg_word_count": avg("rag_word_count"),
    }
