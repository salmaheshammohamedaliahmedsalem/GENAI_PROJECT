"""
llm.py — LLM answer generation with automatic provider selection.

Primary  : Groq API (llama-3.3-70b-versatile) when GROQ_API_KEY is set.
Fallback : Local Qwen/Qwen2.5-0.5B-Instruct via Hugging Face transformers.

Public API
----------
    build_prompt(query, results) -> str
    generate_answer(query, results, max_new_tokens) -> str
"""
from __future__ import annotations

import os
from typing import Optional

from src.rag.mix.config import GROQ_MODEL, LOCAL_MODEL_ID

SYSTEM_INSTRUCTION = (
    "You are a study assistant. A student will give you numbered context passages and a question. "
    "Your job: answer the question using ONLY what the passages say.\n"
    "Rules:\n"
    "- Write in clear, plain English suitable for a student.\n"
    "- If a passage contains a formula, explain in words what each part means — do not just copy the symbols.\n"
    "- After each fact, cite it like this: (passage 2, offline).\n"
    "- If the passages do not contain enough information, reply with exactly: I don't know.\n"
    "- Answer directly and stop. Never generate questions, never ask the student anything, "
    "never add a 'follow-up question' section."
)


def build_prompt(query: str, results: list[dict], max_ctx_chars: int = 4000) -> str:
    """Build a numbered-passage context block for the LLM."""
    context_parts: list[str] = []
    budget = max_ctx_chars
    for rank, r in enumerate(results, 1):
        chunk = r["chunk"]
        text = " ".join((chunk.get("text") or "").split())
        source_type = chunk.get("source_type", "unknown")
        provider = (chunk.get("metadata") or {}).get("provider", source_type)
        page = chunk.get("page")
        source_label = f"{provider}, Page {page}" if page else provider
        snippet = f"[{rank}] (source: {source_label})\n{text}"
        if len(snippet) > budget:
            context_parts.append(snippet[:budget])
            break
        context_parts.append(snippet)
        budget -= len(snippet) + 1

    context_block = "\n\n".join(context_parts) if context_parts else "(no context retrieved)"
    return f"Context passages:\n\n{context_block}\n\nQuestion: {query}"


_groq_client = None
_groq_error = ""


def _get_groq_client():
    global _groq_client, _groq_error
    if _groq_client is not None:
        return _groq_client
    if _groq_error:
        return None
    key = os.getenv("GROQ_API_KEY")
    if not key:
        _groq_error = "GROQ_API_KEY not set"
        return None
    try:
        from openai import OpenAI  # noqa: PLC0415
        _groq_client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        return _groq_client
    except Exception as e:
        _groq_error = f"{type(e).__name__}: {e}"
        return None


def _generate_groq(query: str, results: list[dict], max_new_tokens: int) -> Optional[str]:
    client = _get_groq_client()
    if client is None:
        return None
    try:
        prompt = build_prompt(query, results)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content
    except Exception:
        return None


_local_pipeline = None
_local_error = ""


def _get_local_pipeline():
    global _local_pipeline, _local_error
    if _local_pipeline is not None:
        return _local_pipeline
    if _local_error:
        return None
    try:
        import torch  # noqa: PLC0415
        from transformers import pipeline as hf_pipeline  # noqa: PLC0415

        device = 0 if torch.cuda.is_available() else -1
        print(f"[LLM] Loading {LOCAL_MODEL_ID} on {'GPU' if device == 0 else 'CPU'} …", flush=True)
        _local_pipeline = hf_pipeline(
            "text-generation",
            model=LOCAL_MODEL_ID,
            device=device,
            torch_dtype=torch.float16 if device == 0 else torch.float32,
            trust_remote_code=True,
        )
        print("[LLM] Local model ready.", flush=True)
        return _local_pipeline
    except Exception as e:
        _local_error = f"{type(e).__name__}: {e}"
        print(f"[LLM] WARNING — could not load local model: {_local_error}", flush=True)
        return None


def _generate_local(query: str, results: list[dict], max_new_tokens: int) -> str:
    pipe = _get_local_pipeline()
    if pipe is None:
        return f"[LLM unavailable: {_local_error or 'unknown error'}]"
    prompt = (
        "You are a helpful research assistant. "
        "Answer the question based ONLY on the provided context.\n\n"
        f"### Context\n{build_prompt(query, results)}\n\n"
        "### Answer"
    )
    try:
        output = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=pipe.tokenizer.eos_token_id,
            eos_token_id=pipe.tokenizer.eos_token_id,
        )
        generated: str = output[0]["generated_text"]
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        return generated.strip()
    except Exception as e:
        return f"[Generation error: {type(e).__name__}: {e}]"


def generate_answer(
    query: str,
    results: list[dict],
    max_new_tokens: int = 256,
) -> str:
    """Generate an answer using Groq API if available, otherwise local Qwen."""
    groq_answer = _generate_groq(query, results, max_new_tokens)
    if groq_answer is not None:
        return groq_answer
    return _generate_local(query, results, max_new_tokens)
