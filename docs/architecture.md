# Architecture

GenAI Mentor is an educational multi-agent system for students learning Generative AI. It combines course-grounded retrieval, clear prompt templates, tool use, LoRA fine-tuning evidence, safety checks, and an auditable Streamlit interface.

## Runtime Flow

```text
Student query
  ↓
Safety node
  ↓
Planner/router node
  ├─ offline_only → course BM25/optional semantic retrieval
  ├─ online_only  → approved external retrieval when configured
  ├─ hybrid       → course + approved external retrieval
  ├─ tool_only    → calculator, quiz, or grader
  └─ no_retrieval → safe refusal or direct non-grounded response
  ↓
Response node
  ├─ TutorAgent
  ├─ QuizAgent
  ├─ GraderAgent
  └─ tool node
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
| RAG layer | `src/rag/` | Retrieves course chunks locally with BM25 and optional semantic/online extensions. |
| Tutor agent | `src/agents/tutor_agent.py` | Produces student-friendly grounded explanations. |
| Quiz agent | `src/agents/quiz_agent.py` | Generates practice questions with answers and explanations. |
| Grader agent | `src/agents/grader_agent.py` | Scores student answers and returns targeted feedback. |
| Checker agent | `src/agents/checker_agent.py` | Checks grounding and citation validity. |
| Tool layer | `src/tools/` | Provides calculator, quiz, grading, citations, and progress helpers. |
| Fine-tuning | `notebooks/03_finetuning_complete.ipynb`, `src/finetuning/` | Trains and evaluates a Qwen LoRA adapter for educational behavior. |
| GUI | `app.py` | Presents the student learning flow, agents/prompts, RAG inspector, traces, fine-tuning, evaluation, safety, and run checks. |

## Student Interface

The Streamlit app is organized around how a student uses the system:

1. **Overview:** Shows project evidence and the learning flow.
2. **Learn & Practice:** Main chat for explanations, quizzes, grading, and tool-backed answers.
3. **Agents & Prompts:** Shows the graph nodes, graph edges, and exact prompt templates.
4. **Evidence/RAG:** Lets reviewers inspect retrieved chunks before answer generation.
5. **Agent Trace:** Shows the saved JSON trace for auditability.
6. **Fine-Tuning:** Shows dataset splits, Qwen LoRA adapter status, metrics, and output files.
7. **Evaluation:** Shows evaluation summaries and result tables.
8. **Safety:** Demonstrates academic-integrity refusals.
9. **Run & Check:** Runs tests, index build, evaluation, and bounded fine-tuning smoke checks from the UI.
