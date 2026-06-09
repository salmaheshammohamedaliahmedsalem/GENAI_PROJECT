# 🚀 Adaptive Multi-Agent GenAI Tutor with Hybrid RAG

Welcome to the working **Adaptive Multi-Agent GenAI Tutor with Hybrid Online/Offline RAG**.

This system ingests academic lecture PDFs, builds course-grounded retrieval indexes, fine-tunes a Qwen LoRA adapter for educational behavior, and exposes a Streamlit student tutor with backend trace visibility. The fine-tuning data teaches the distinct behavioral formats required to act as an educational **Tutor**, an academic **Examiner**, and a reflective **Critic**.

---

## 🌐 Live Demo

Open the deployed Streamlit app: [StudyBuddy GenAI](https://studybuddygenai.streamlit.app/)

---

## 🎯 Fine-Tuning Philosophy

Unlike naive QA generation, our dataset preparation is structured around a core project thesis:
> **RAG (Retrieval-Augmented Generation)** is used to inject factual course knowledge at inference time, while **Fine-Tuning** is used to sculpt the behavioral alignment, tone, and cognitive structuring of our multi-agent personas.

By separating **knowledge (RAG)** from **behavior (fine-tuning)**, we achieve a highly stable, highly intelligent, and structured educational ecosystem.

---

## 📂 Project Directory Structure

```text
genai_project/
├── data/
│   ├── raw/course_pdfs/       # Raw source lecture PDFs (LLM Lectures 1 - 9)
│   ├── extracted_text/        # Normalized page-by-page JSONL extractions
│   ├── chunks/                # Word-bounded slide/page text chunks
│   ├── finetune/              # Chat-format SFT data and train/val/test splits
│   │   ├── tutor_dataset.jsonl
│   │   ├── examiner_dataset.jsonl
│   │   ├── critic_dataset.jsonl
│   │   ├── combined_dataset.jsonl
│   │   ├── tutor_dataset_clean.jsonl
│   │   ├── examiner_dataset_clean.jsonl
│   │   ├── critic_dataset_clean.jsonl
│   │   └── combined_dataset_clean.jsonl
│   ├── metadata/              # Generation summary CSV sheets
│   │   └── chunk_summary.csv
│   ├── processed/             # BM25 index and processed retrieval artifacts
│   └── vector_db/             # Optional semantic vector store
├── notebooks/
│   ├── 01_dataset_generation_pipeline.ipynb  # Dataset generation walkthrough
│   └── 03_finetuning_complete.ipynb          # Qwen LoRA train/evaluate/test workflow
├── src/
│   ├── agents/                # Safety, planner, adaptation, tutor, quiz, grader, checker, graph
│   ├── evaluation/            # Retrieval, answer, safety, and baseline evaluation
│   ├── finetuning/            # LoRA training, inference, and base-vs-tuned comparison
│   ├── ingestion/             # PDF/document loading, chunking, and index building
│   ├── llm/                   # OpenAI/Groq/local/Qwen model selection and prompts
│   ├── rag/                   # Offline BM25, optional semantic, online, hybrid retrieval
│   ├── tools/                 # Calculator, quiz, grading, citation, progress tools
│   ├── extract_pdf_text.py    # Lecture PDF text extractor
│   ├── chunk_lectures.py      # Lecture chunk generator
│   ├── generate_synthetic_dataset.py # API/procedural SFT dataset synthesizer
│   └── validate_dataset.py    # Formatting, schema, and deduplication validator
├── app.py                     # Streamlit Student + Backend Tracking GUI
├── docs/                      # Architecture, demo, evaluation, ethics, report outline
├── outputs/                   # Fine-tuning, evaluation, and agent trace artifacts
└── README.md                  # Project overview & running instructions
```

---

## 🛠️ Required Packages & Installation

The app uses Streamlit, LangGraph, BM25 retrieval, optional hosted LLM APIs, and a deterministic local fallback. The dataset pipeline also uses PDF extraction utilities and can generate examples with OpenAI/Groq APIs or procedural templates.

```bash
pip install -r requirements.txt
```

For Streamlit Community Cloud, use the default `requirements.txt`. It intentionally excludes ChromaDB and fine-tuning libraries so the deployed app can start reliably with BM25 retrieval, online retrieval, tools, agents, evaluation displays, and fine-tuning artifacts. For local semantic retrieval, install `requirements_semantic.txt`. For local LoRA training/inference, install `requirements_finetune.txt`. A LoRA adapter is not a standalone model: it also needs its base model weights, such as `Qwen/Qwen2.5-0.5B-Instruct`, cached locally or allowed to download with `LOCAL_MODEL_ALLOW_DOWNLOADS=true`.

---

## 🔑 Environment Variables Setup

Create a `.env` file at the root of the project to set up hosted LLMs and retrieval APIs. The app still runs without keys using the deterministic local fallback.

```ini
# Hosted answer generation and optional dataset generation
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Local fine-tuned model inference
LOCAL_MODEL_ALLOW_DOWNLOADS=false

# Online retrieval
TAVILY_API_KEY=your_tavily_api_key_here
```

> [!TIP]
> **Zero-Dependency Fallback Engine**: If no API keys are present in your environment, the pipeline automatically redirects generation requests to our robust, in-memory procedural template generator. This constructs hundreds of high-fidelity, structured examples, guaranteeing that the pipeline is runnable out-of-the-box in any sandbox.

---

## 🏃 How to Run the Dataset Pipeline

To run the pipeline from the project root, execute the following commands in sequence:

### 1. Extract Text from Lecture Slides
Extracts page-by-page text, normalizes characters, and saves individual slide files.
```bash
python3 src/extract_pdf_text.py
```

### 2. Chunk and Categorize Pages
Groups text and automatically predicts topic categories using our keyword-based classifiers.
```bash
python3 src/chunk_lectures.py
```

### 3. Synthesize Personas Datasets
Runs the synthetic dataset generator to create Tutor (Analogy/Misconception schema), Examiner (MCQ/Grading schema), and Critic (Reflection/Groundedness schema) datasets.
```bash
python3 src/generate_synthetic_dataset.py
```

### 4. Validate and Clean Datasets
Checks target formats, verifies that all five required tutoring blocks are present, removes duplicates, and prints validation reports.
```bash
python3 src/validate_dataset.py
```

---

## 📊 Outputs & Fine-Tuning Targets

Our pipeline successfully synthesizes and cleans the following target datasets:

| Persona Dataset | SFT Output Formatting Schema | Clean Target Size |
| :--- | :--- | :--- |
| **Tutor Dataset** | `Simple explanation` \| `Analogy` \| `Course-grounded answer` \| `Common misconception` \| `Quick check question` | **2,656 examples** |
| **Examiner Dataset** | `Question` \| `Choices` \| `Correct answer` OR `Score` \| `Feedback` \| `Corrected answer` | **1,992 examples** |
| **Critic Dataset** | `Critique` \| `Groundedness` \| `Missing points` \| `Improved answer` | **1,328 examples** |
| **Combined Dataset** | Aggregated instruction-tuning data containing all three agent modalities | **5,976 examples** |

---

## ✅ Current LoRA, RAG, and Agent Status

The end-to-end system is implemented:

1. **LoRA SFT Alignment**: `Qwen/Qwen2.5-0.5B-Instruct` has been fine-tuned with PEFT/LoRA on Apple MPS. Final adapter artifacts are in `outputs/finetune/qwen_0_5b_lora_adapter_salma/`.
2. **Offline/Online RAG Integration**: BM25 retrieval is built from `data/chunks/lecture_chunks.jsonl`; approved online retrieval uses Tavily when configured and the maintained `ddgs` package as a no-key fallback.
3. **Multi-Agent Orchestration**: Safety, planning, student adaptation, retrieval, tutoring, quiz, grading, checking, and trace-writing nodes are connected through `src/agents/graph.py` using LangGraph when installed.
4. **Prompt Template Layer**: `src/llm/prompts.py` documents the base, router, tutor, quiz, grading, checker, and safety prompts with required inputs.
5. **Model Selection**: The Student GUI shows canonical Salma, Fatma, and Khadija fine-tuned models, plus base Qwen, Groq-hosted chat, OpenAI-hosted chat, or deterministic fallback depending on installed packages, configured keys, and local base-model cache/download availability. Backend Tracking lists every saved adapter/experiment.
6. **GUI Showcase**: `app.py` is split into two primary modes: **Student** for level-adaptive chat plus retrieved-content review, and **Backend Tracking** for architecture, agents/prompts, RAG diagnostics, traces, fine-tuning, evaluation, safety, and run checks.
