# Architecture

GenAI Mentor is an educational multi-agent system for students learning Generative AI. It combines course-grounded retrieval, clear prompt templates, tool use, LoRA fine-tuning evidence, safety checks, and an auditable Streamlit interface.

## Runtime Flow

```text
Student query
  ↓
Safety node
  ↓
Planner/router node
  ↓
Student adaptation node
  ├─ beginner     → simple definitions, analogy, quick check
  ├─ intermediate → course terminology, mechanisms, implementation steps
  └─ advanced     → tradeoffs, assumptions, failure modes, evaluation
  ├─ offline_only → course BM25/optional semantic retrieval
  ├─ online_only  → approved external retrieval with Tavily or ddgs fallback
  ├─ hybrid       → course + approved external retrieval
  ├─ tool_only    → calculator, quiz, or grader
  └─ no_retrieval → safe refusal or direct non-grounded response
  ↓
Response node
  ├─ TutorAgent
  ├─ QuizAgent
  ├─ GraderAgent
  └─ tool node
  └─ model selector
      ├─ Qwen LoRA adapter
      ├─ base Qwen model
      ├─ Groq-hosted chat
      ├─ OpenAI-hosted chat
      └─ deterministic fallback
  ↓
Checker node
  ↓
Trace writer
  ↓
Student-facing answer + sources + trace
```

## Implemented Components

| Component | Implementation | Purpose |
| --- | --- | --- |
| Prompt templates | `src/llm/prompts.py` | Defines base, router, tutor, quiz, grading, checker, and safety prompts with documented inputs. |
| LangGraph workflow | `src/agents/graph.py` | Builds a `StateGraph` when `langgraph` is installed; otherwise runs the same nodes sequentially so local demos remain functional. |
| Safety agent | `src/agents/safety_agent.py` | Blocks cheating, plagiarism, hidden exam answer requests, and policy bypass attempts. |
| Planner agent | `src/agents/planner_agent.py` | Selects retrieval mode and whether tools, quizzes, or grading are needed. |
| Student adaptation agent | `src/agents/adaptation_agent.py` | Adapts answer depth, terminology, examples, and quiz difficulty to beginner, intermediate, or advanced students. |
| LLM/model selector | `src/llm/model_registry.py`, `src/llm/client.py` | Selects the fine-tuned Qwen LoRA adapter, base Qwen model, Groq-hosted chat, OpenAI-hosted chat, or deterministic fallback based on available dependencies and API keys. |
| RAG layer | `src/rag/` | Retrieves course chunks locally with BM25 and approved online results with Tavily or `ddgs` fallback. |
| Tutor agent | `src/agents/tutor_agent.py` | Produces student-friendly grounded explanations. |
| Quiz agent | `src/agents/quiz_agent.py` | Generates practice questions with answers and explanations. |
| Grader agent | `src/agents/grader_agent.py` | Scores student answers and returns targeted feedback. |
| Checker agent | `src/agents/checker_agent.py` | Checks grounding and citation validity. |
| Tool layer | `src/tools/` | Provides calculator, quiz, grading, citations, and progress helpers. |
| Fine-tuning | `notebooks/03_finetuning_complete.ipynb`, `src/finetuning/` | Trains and evaluates a Qwen LoRA adapter for educational behavior. |
| GUI | `app.py` | Presents a Student mode for chat plus retrieved content, and a Backend Tracking mode for implementation evidence, traces, fine-tuning, evaluation, safety, and run checks. |

## Student Interface

The Streamlit app is split into two top-level buttons:

1. **Student:** Main student chat with a level selector. The left side contains study controls, the center is the conversation, and the right side shows retrieved chunks/sources used for the latest answer.
2. **Backend Tracking:** Reviewer/developer dashboard with Overview, Agents & Prompts, Evidence/RAG, Agent Trace, Fine-Tuning, Evaluation, Safety, and Run & Check tabs.

## Runtime Model Behavior

- The selected model receives the final tutor prompt, including the student profile and retrieved RAG context assembled by `TutorAgent`.
- The recommended local option is the trained `Qwen/Qwen2.5-0.5B-Instruct` LoRA adapter when local fine-tuning dependencies are installed and the required base model is cached or allowed through `LOCAL_MODEL_ALLOW_DOWNLOADS=true` plus `LOCAL_MODEL_DOWNLOAD_ALLOWLIST`.
- Groq is supported as a fast hosted base chatbot through `GROQ_API_KEY` and `GROQ_MODEL`, using the OpenAI-compatible endpoint.
- OpenAI is supported through `OPENAI_API_KEY` and `CHAT_MODEL`.
- If no hosted API or local neural model is available, the app falls back to `LocalRuleBasedLLM` so demos still run without secrets.
