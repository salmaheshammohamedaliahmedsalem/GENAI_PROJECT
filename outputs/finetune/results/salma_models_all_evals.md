# Salma Models - Complete Evaluation Report

Generated from local project artifacts on 2026-06-09 19:22:24 UTC.

## Scope

This report includes every evaluation artifact found for Salma-named saved adapters in this project:

| Adapter | Base model | Evaluation coverage |
| --- | --- | --- |
| `qwen_0_5b_lora_adapter_salma` | `Qwen/Qwen2.5-0.5B-Instruct` | Full training metrics, validation loss history, and 25-example base-vs-tuned comparison |
| `lora_adapter_salma` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Notebook smoke/structure checks only |

Note: `lora_adapter_salma` is a TinyLlama smoke-test adapter, not the final recommended Salma model. The final recommended Salma model is `qwen_0_5b_lora_adapter_salma`.

## Source Artifacts Used

- `outputs/finetune/training_log.json`
- `outputs/finetune/train_metrics.json`
- `outputs/finetune/eval_metrics.json`
- `outputs/finetune/results/evaluation_summary.json`
- `outputs/finetune/results/base_vs_tuned_comparison.csv`
- `outputs/finetune/notebook_adapter_eval.csv`
- `outputs/finetune/qwen_0_5b_lora_adapter_salma/adapter_config.json`
- `outputs/finetune/lora_adapter_salma/adapter_config.json`

## Final Salma Model: `qwen_0_5b_lora_adapter_salma`

### Training Configuration

| Field | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Device | `mps` |
| Status | `completed` |
| Raw SFT examples | `5976` |
| Clean examples used | `1000` |
| Train / validation / test | `800` / `100` / `100` |
| Split | `80/10/10` |
| Epochs | `2.0` |
| LoRA rank `r` | `8` |
| LoRA alpha | `16` |
| LoRA dropout | `0.05` |
| Target modules | `k_proj, q_proj, down_proj, gate_proj, o_proj, up_proj, v_proj` |

### Final Training and Validation Metrics

| Metric | Value |
| --- | ---: |
| Train loss | 2.325709 |
| Eval loss | 2.198895 |
| Train runtime seconds | 1358.7863 |
| Eval runtime seconds | 18.1280 |
| Train samples/sec | 1.1780 |
| Eval samples/sec | 5.5160 |
| Train steps/sec | 0.1470 |
| Eval steps/sec | 5.5160 |

### Validation Loss Over Training

| Step | Epoch | Eval loss | Eval runtime seconds |
| ---: | ---: | ---: | ---: |
| 25 | 0.25 | 2.717208 | 146.8158 |
| 50 | 0.5 | 2.578454 | 18.3045 |
| 75 | 0.75 | 2.461585 | 17.8267 |
| 100 | 1.0 | 2.367511 | 17.6241 |
| 125 | 1.25 | 2.295518 | 17.9525 |
| 150 | 1.5 | 2.241843 | 17.7004 |
| 175 | 1.75 | 2.209228 | 17.8441 |
| 200 | 2.0 | 2.198895 | 17.6730 |
| 200 | 2.0 | 2.198895 | 18.1280 |

### Base-vs-Tuned Evaluation Summary

| Metric | Value |
| --- | ---: |
| Evaluation examples | 25 |
| Average base score | 0.423 |
| Average tuned score | 0.484 |
| Average improvement | 0.061 |
| Tuned wins | 14 |
| Base wins or ties | 11 |

Interpretation: Salma Qwen LoRA improved the average rubric score from `0.423` to `0.484`, winning `14` of `25` examples. It is better overall, but not uniformly better on every prompt.

### Scores by Instruction Type

| Instruction | Examples | Avg base | Avg tuned | Avg delta | Tuned wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Critique and improve the assistant answer using the lecture context. | 3 | 0.521 | 0.399 | -0.122 | 0 |
| Generate an exam-style question from the lecture context. | 7 | 0.243 | 0.477 | 0.234 | 6 |
| Grade the student's answer using the lecture context. | 3 | 0.589 | 0.616 | 0.027 | 1 |
| Teach the concept using the required tutoring format. | 12 | 0.463 | 0.477 | 0.014 | 7 |

### All 25 Base-vs-Tuned Evaluation Rows

This table includes every scored comparison row. Full generated text is preserved in the source CSV listed above.

| # | Instruction | Base score | Tuned score | Delta | Winner | Base hallucination | Tuned hallucination | Base length ok | Tuned length ok | Notes | Input preview |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | Generate an exam-style question from the lecture context. | 0.477 | 0.511 | 0.034 | Tuned | False | False | True | True | Tuned score higher | Topic: LLM Alignment (RLHF / DPO)<br>Difficulty: medium<br>Lecture context: 58<br>Evaluate the human-aligned LLM<br>Evaluate the human-aligned LLM<br>Summarization<br>Dataset<br>Evaluate using the toxicity score<br>Toxicity score before:<br>0.14<br>I... |
| 2 | Generate an exam-style question from the lecture context. | 0.207 | 0.580 | 0.373 | Tuned | False | False | False | True | Tuned score higher | Topic: Retrieval-Augmented Generation (RAG)<br>Difficulty: medium<br>Lecture context: Keyword search - TF-IDF<br>27<br>Term Frequency – Inverse Document Frequency |
| 3 | Teach the concept using the required tutoring format. | 0.590 | 0.656 | 0.066 | Tuned | False | False | True | True | Tuned score higher | Student level: advanced<br>Question: What is Low-Rank Adaptation (LoRA) and how does it optimize fine-tuning?<br>Retrieved lecture context: Fine-tuning,<br>Instruction<br>Tuning and PERF<br>Dr. Tamer Arafa<br>1 |
| 4 | Critique and improve the assistant answer using the lecture context. | 0.460 | 0.340 | -0.120 | Base/tie | False | False | True | True | Base score equal or higher | Question: What are the best practices for structuring system prompts?<br>Lecture context: Summaryofin-contextlearning(ICL)<br>Classify this review:<br>I loved this movie!<br>Sentiment:<br>Prompt // Zero Shot<br>Classify this review:<br>I lov... |
| 5 | Teach the concept using the required tutoring format. | 0.445 | 0.367 | -0.078 | Base/tie | False | False | True | True | Base score equal or higher | Student level: beginner<br>Question: What are the best practices for structuring system prompts?<br>Retrieved lecture context: Return on investment on prompt engineering<br>Performance<br>Time spent prompt<br>engineering<br>Reflection wit... |
| 6 | Critique and improve the assistant answer using the lecture context. | 0.578 | 0.347 | -0.231 | Base/tie | False | False | True | True | Base score equal or higher | Question: Explain the mathematics of the scaled dot-product attention.<br>Lecture context: Representation Models<br>17<br>Bidirectional Encoder<br>Representations from<br>Transformers (BERT)<br>classification<br>token<br>Assistant answer: Trans... |
| 7 | Teach the concept using the required tutoring format. | 0.564 | 0.622 | 0.058 | Tuned | False | False | True | True | Tuned score higher | Student level: intermediate<br>Question: What are the main components of a RAG pipeline?<br>Retrieved lecture context: Understanding Embedding Models<br>No simple interpretation of X<br>and Y axis<br>…instead…<br>points “float around” in<br>... |
| 8 | Teach the concept using the required tutoring format. | 0.548 | 0.454 | -0.094 | Base/tie | False | False | True | True | Base score equal or higher | Student level: beginner<br>Question: Explain the difference between Chain of Thought (CoT) and standard prompting.<br>Retrieved lecture context: Contrastive Language-Image Pretraining (CLIP)<br>●<br>CLIP differs from traditional vis... |
| 9 | Grade the student's answer using the lecture context. | 0.627 | 0.611 | -0.016 | Base/tie | False | False | True | True | Base score equal or higher | Question: Explain the difference between RAG and fine-tuning.<br>Student answer: RAG is when you train a model on search engine results so it learns everything online.<br>Lecture context: More challenging: Customer service age... |
| 10 | Generate an exam-style question from the lecture context. | 0.192 | 0.507 | 0.315 | Tuned | False | False | False | True | Tuned score higher | Topic: LLM Agents & Multi-Agent Systems<br>Difficulty: medium<br>Lecture context: Successful task completion rate (%)<br>[Adapted from “Executable Code actions Elicit Better LLM Agents”, Wang et al. 2024]<br>Planning with code impro... |
| 11 | Teach the concept using the required tutoring format. | 0.026 | 0.026 | 0.000 | Base/tie | False | False | False | False | Base score equal or higher | Student level: advanced<br>Question: Explain the difference between RAG and fine-tuning.<br>Retrieved lecture context: Prompt Template<br># System Instructions<br>You are an useful assistant for geographic information . Only use ret... |
| 12 | Teach the concept using the required tutoring format. | 0.440 | 0.346 | -0.094 | Base/tie | False | False | True | True | Base score equal or higher | Student level: intermediate<br>Question: Explain the difference between Chain of Thought (CoT) and standard prompting.<br>Retrieved lecture context: Multi-task, instruction ﬁne-tuning<br>Pre-trained<br>LLM<br>Model<br>Instruction ﬁne-tune... |
| 13 | Teach the concept using the required tutoring format. | 0.541 | 0.484 | -0.057 | Base/tie | False | False | True | True | Base score equal or higher | Student level: advanced<br>Question: What is the difference between small and large language models?<br>Retrieved lecture context: Introduction to<br>Agentic Workflows<br>Task decomposition:<br>Identifying the steps in<br>a workflow<br>17 |
| 14 | Generate an exam-style question from the lecture context. | 0.407 | 0.277 | -0.130 | Base/tie | False | False | True | False | Base score equal or higher | Topic: Retrieval-Augmented Generation (RAG)<br>Difficulty: medium<br>Lecture context: Having an LLM initiate a clothing return<br>ShopBot<br>Ok, I’ve found your order.<br>Do you want to return<br>any other items from that<br>order?<br>No, only ... |
| 15 | Grade the student's answer using the lecture context. | 0.627 | 0.606 | -0.021 | Base/tie | False | False | True | True | Base score equal or higher | Question: What are the best practices for structuring system prompts?<br>Student answer: Prompt engineering is just typing questions in ChatGPT until you get the right answer.<br>Lecture context: Traditional Language Model Usa... |
| 16 | Generate an exam-style question from the lecture context. | 0.091 | 0.367 | 0.276 | Tuned | False | False | False | True | Tuned score higher | Topic: LLM Agents & Multi-Agent Systems<br>Difficulty: medium<br>Lecture context: Example: Marketing team<br>researcher<br>writer<br>Researcher<br>Tasks<br>•<br>Analyze market trends<br>•<br>Research competitors<br>Tools<br>•<br>Web search<br>Graphic designer<br>Ta... |
| 17 | Critique and improve the assistant answer using the lecture context. | 0.526 | 0.511 | -0.015 | Base/tie | False | False | True | True | Base score equal or higher | Question: Explain the difference between Chain of Thought (CoT) and standard prompting.<br>Lecture context: Bag of words<br>Bag of Words<br>“Making pizza without a pizza oven”<br>Word order is ignored, only word presence and frequen... |
| 18 | Grade the student's answer using the lecture context. | 0.512 | 0.630 | 0.118 | Tuned | False | False | True | True | Tuned score higher | Question: What is the difference between small and large language models?<br>Student answer: An LLM is a giant database of sentences that it searches through when you ask it a question.<br>Lecture context: Lecture References<br>●... |
| 19 | Generate an exam-style question from the lecture context. | 0.312 | 0.736 | 0.424 | Tuned | False | False | False | True | Tuned score higher | Topic: Large Language Models Fundamentals<br>Difficulty: medium<br>Lecture context: GenerativeAI projectlifecycle<br>56 |
| 20 | Teach the concept using the required tutoring format. | 0.440 | 0.532 | 0.092 | Tuned | False | False | True | True | Tuned score higher | Student level: advanced<br>Question: Explain causal language modeling and how it differs from masked language modeling.<br>Retrieved lecture context: Instruction following results (GPT-5)<br>Identified PII (type → value):<br>1.Full ... |
| 21 | Generate an exam-style question from the lecture context. | 0.015 | 0.360 | 0.345 | Tuned | False | False | False | True | Tuned score higher | Topic: LLM Evaluation & Safety<br>Difficulty: hard<br>Lecture context: LLM Evaluation - Metrics - ROUGE-2<br>Generated output:<br>It is very cold outside.<br>It is<br>is very<br>very cold<br>cold outside<br>Reference (human):<br>It is cold outside.<br>I... |
| 22 | Teach the concept using the required tutoring format. | 0.521 | 0.678 | 0.157 | Tuned | False | False | True | True | Tuned score higher | Student level: beginner<br>Question: Explain how autoregressive decoding works in LLM inference.<br>Retrieved lecture context: What Is Language AI?<br>●<br>Language AI refers to a subfield of AI that focuses on developing technologi... |
| 23 | Teach the concept using the required tutoring format. | 0.463 | 0.487 | 0.024 | Tuned | False | False | True | True | Tuned score higher | Student level: beginner<br>Question: How do agents utilize tools in a ReAct loop?<br>Retrieved lecture context: Building up the ReAct prompt<br>Question<br>Thought<br>Action<br>Observation<br>Question<br>Thought<br>Action<br>Observation<br>ReAct example... |
| 24 | Teach the concept using the required tutoring format. | 0.410 | 0.488 | 0.078 | Tuned | False | False | True | True | Tuned score higher | Student level: intermediate<br>Question: Explain the MMLU benchmark and its significance.<br>Retrieved lecture context: LLM Evaluation - Metrics - ROUGE-2<br>Generated output:<br>It is very cold outside.<br>bigram matches<br><br>bigrams in r... |
| 25 | Teach the concept using the required tutoring format. | 0.566 | 0.580 | 0.014 | Tuned | False | False | True | True | Tuned score higher | Student level: beginner<br>Question: What is the difference between small and large language models?<br>Retrieved lecture context: Mixture of Experts (MOE)<br>40 |

## Salma Smoke Adapter: `lora_adapter_salma`

### Adapter Configuration

| Field | Value |
| --- | --- |
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| LoRA rank `r` | `8` |
| LoRA alpha | `16` |
| LoRA dropout | `0.05` |
| Target modules | `v_proj, q_proj` |

### Notebook Structure Evaluation

| Mode | Required phrases | Found phrases | Structure score | Found phrase list | Output preview |
| --- | ---: | ---: | ---: | --- | --- |
| tutor | 5 | 0 | 0.000 |  | Teacher: Sure, I'd be happy to explain LoRA fine-tuning. LoRA (Low-Rank Approximate Regression) is a technique used in machine learning to improve the accuracy of regression models. It works by reducing the dimensionality of the data while preserving the relationships between the variables.<br><br>Student: Okay, so what does LoRA fine-tuning do?<br><br>Teacher: LoRA fine-tuning is a process that involves reducing the dimensionality of the data while preserving the relationships between the variables. It doe |
| examiner | 3 | 0 | 0.000 |  | Grade: Pass<br><br>The student's answer is correct. RAG (Reinforcement Aging Gradient) is a technique used in GenAI to train the model to learn from past interactions with users. The model is retrained every time a user asks a question, which helps it to better understand the user's intent and respond accordingly. This technique is an important part of GenAI's ability to adapt to changing user needs and preferences. |
| critic | 4 | 0 | 0.000 |  | Moderator: Thank you for your response. Can you please provide a revised version of the assistant answer that addresses the instruction to improve the structure?<br><br>Assistant answer: Prompts do not matter because the model already knows everything.<br><br>Moderator: Thank you for your response. Can you please provide a revised version of the assistant answer that includes a specific example of how prompts can affect LLM behavior?<br><br>Assistant answer: Prompts can affect LLM behavior in the following ways:<br> |

Interpretation: the TinyLlama smoke adapter did not satisfy the required tutor/examiner/critic output structures in the notebook check. It should remain a smoke-test artifact, not the recommended Student model.

## Bottom Line

- Recommended Salma model: `qwen_0_5b_lora_adapter_salma`.
- It improved average evaluation score by `0.061` over the base model and won `14/25` comparisons.
- TinyLlama `lora_adapter_salma` is retained for evidence/smoke testing, but its structure evaluation scored `0.000` in all three tested modes.
- In Student mode, only the canonical Salma model should be shown; the full adapter list belongs in Backend Tracking.
