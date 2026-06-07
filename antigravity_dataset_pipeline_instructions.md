# Antigravity Instructions — GenAI Project Fine-Tuning Dataset Pipeline

## Current project state

The project folder is:

genai_project/
└── data/
    └── raw_pdfs/
        ├── LLM Lecture 1.pdf
        ├── LLM Lecture 2.pdf
        ├── LLM Lecture 3.pdf
        ├── LLM Lecture 4.pdf
        ├── LLM Lecture 5 (1).pdf
        ├── LLM Lecture 6 (1).pdf
        ├── LLM Lecture 7 (1).pdf
        ├── LLM Lecture 8 (1).pdf
        └── LLM Lecture 9 (1).pdf

We want to start with the fine-tuning preparation stage.

The goal is NOT to fine-tune immediately. First, build a clean data pipeline that turns the lecture PDFs into structured instruction-tuning datasets for an out-of-the-box GenAI educational assistant.

## Main objective

Build a dataset generation pipeline for a project called:

Adaptive Multi-Agent GenAI Tutor with Hybrid Online/Offline RAG

The assistant should eventually support:
1. Tutor mode
2. Examiner mode
3. Critic/reflection mode
4. Hybrid offline + online RAG

For now, implement the offline lecture processing and fine-tuning dataset preparation.

---

# Required folder structure

Inside `genai_project`, create:

data/
├── raw_pdfs/
├── extracted_text/
├── chunks/
├── finetuning/
│   ├── tutor_dataset.jsonl
│   ├── examiner_dataset.jsonl
│   ├── critic_dataset.jsonl
│   └── combined_dataset.jsonl
├── rag_db/
└── metadata/

src/
├── extract_pdf_text.py
├── chunk_lectures.py
├── generate_synthetic_dataset.py
├── validate_dataset.py
└── utils.py

notebooks/
└── 01_dataset_generation_pipeline.ipynb

---

# Step 1 — PDF text extraction

Create `src/extract_pdf_text.py`.

Use PyMuPDF (`fitz`) to extract text from every PDF in:

data/raw_pdfs/

For each page, save metadata:

{
  "lecture_file": "LLM Lecture 7 (1).pdf",
  "lecture_id": "lecture_7",
  "page": 14,
  "text": "...",
  "char_count": 1234
}

Save one JSONL file per lecture in:

data/extracted_text/

Example:

data/extracted_text/lecture_7_pages.jsonl

Also create a combined file:

data/extracted_text/all_lectures_pages.jsonl

Important:
- Skip empty pages only if text is completely empty.
- Keep page numbers.
- Normalize whitespace.
- Do not remove technical terms.
- Keep equations and bullet points if extracted.

---

# Step 2 — Chunk lectures by slide/page

Create `src/chunk_lectures.py`.

Chunking strategy:
- Prefer one chunk per slide/page.
- If a page is very long, split into smaller chunks of around 700–1000 words.
- Preserve metadata.

Each chunk should look like:

{
  "chunk_id": "lecture_7_p14_c1",
  "lecture_id": "lecture_7",
  "lecture_file": "LLM Lecture 7 (1).pdf",
  "page": 14,
  "topic_guess": "RAG / Retrieval / Vector Search / Agentic AI etc.",
  "text": "...",
  "source": "offline_lecture"
}

Save:

data/chunks/lecture_chunks.jsonl

Also create a summary CSV:

data/metadata/chunk_summary.csv

with:
- chunk_id
- lecture_id
- page
- word_count
- topic_guess

---

# Step 3 — Generate synthetic instruction tuning data

Create `src/generate_synthetic_dataset.py`.

This script should read:

data/chunks/lecture_chunks.jsonl

Then generate three datasets.

Important:
Use an LLM API only if an API key is available. If not available, create a fallback template-based generator so the pipeline still runs.

Use environment variables:
- OPENAI_API_KEY, or
- GROQ_API_KEY

If using Groq, use OpenAI-compatible client with:
base_url="https://api.groq.com/openai/v1"

Suggested model:
- llama-3.1-8b-instant or any available Groq model

---

## Dataset A — Tutor Dataset

Purpose:
Fine-tune a model to behave like a structured educational tutor.

For each good lecture chunk, create 2–3 examples.

Format JSONL:

{
  "mode": "tutor",
  "instruction": "Teach the concept using the required tutoring format.",
  "input": "Student level: beginner\nQuestion: What is RAG?\nRetrieved lecture context: ...",
  "output": "Simple explanation: ...\nAnalogy: ...\nCourse-grounded answer: ...\nCommon misconception: ...\nQuick check question: ..."
}

The output MUST always include:

1. Simple explanation
2. Analogy
3. Course-grounded answer
4. Common misconception
5. Quick check question

Save to:

data/finetuning/tutor_dataset.jsonl

Target size:
At least 300 examples if possible.

---

## Dataset B — Examiner Dataset

Purpose:
Fine-tune a model to generate quizzes and grade student answers.

For each concept chunk, create examples for:
- MCQ generation
- true/false generation
- short-answer question generation
- grading a student answer

Format example:

{
  "mode": "examiner",
  "instruction": "Generate an exam-style question from the lecture context.",
  "input": "Topic: RAG\nDifficulty: medium\nLecture context: ...",
  "output": "Question: ...\nChoices: A)... B)... C)... D)...\nCorrect answer: ...\nExplanation: ..."
}

For grading:

{
  "mode": "examiner",
  "instruction": "Grade the student's answer using the lecture context.",
  "input": "Question: What is RAG?\nStudent answer: RAG retrains the model.\nLecture context: ...",
  "output": "Score: 1/5\nFeedback: RAG does not retrain the model. It retrieves relevant context and injects it into the prompt.\nCorrected answer: ..."
}

Save to:

data/finetuning/examiner_dataset.jsonl

Target size:
At least 200 examples.

---

## Dataset C — Critic / Reflection Dataset

Purpose:
Fine-tune a critic model or adapter that checks answer quality.

Create examples where the input includes:
- question
- lecture context
- weak/bad answer

The output should give:
- problems found
- whether answer is grounded
- missing concepts
- improved answer

Format:

{
  "mode": "critic",
  "instruction": "Critique and improve the assistant answer using the lecture context.",
  "input": "Question: ...\nLecture context: ...\nAssistant answer: ...",
  "output": "Critique: ...\nGroundedness: grounded / partially grounded / not grounded\nMissing points: ...\nImproved answer: ..."
}

Save to:

data/finetuning/critic_dataset.jsonl

Target size:
At least 150 examples.

---

# Step 4 — Combine datasets

Create:

data/finetuning/combined_dataset.jsonl

Each row should have:
- mode
- instruction
- input
- output
- source_chunk_id
- lecture_id
- page

---

# Step 5 — Dataset validation

Create `src/validate_dataset.py`.

It should check:

1. No empty instruction/input/output.
2. Every example has mode.
3. Tutor outputs contain all five required sections.
4. Examiner examples contain either question/answer or score/feedback.
5. Critic examples contain critique and improved answer.
6. Remove duplicates.
7. Print dataset statistics.

Output:
- number of examples by mode
- average input length
- average output length
- number of invalid rows
- saved cleaned files

Create cleaned versions:

data/finetuning/tutor_dataset_clean.jsonl
data/finetuning/examiner_dataset_clean.jsonl
data/finetuning/critic_dataset_clean.jsonl
data/finetuning/combined_dataset_clean.jsonl

---

# Step 6 — Notebook

Create:

notebooks/01_dataset_generation_pipeline.ipynb

The notebook should explain and run:

1. Extract text from PDFs
2. Show sample extracted pages
3. Chunk lectures
4. Show sample chunks
5. Generate synthetic fine-tuning data
6. Validate dataset
7. Show dataset statistics
8. Display 3 examples from each dataset

Use markdown explanations because this will be shown in the project.

---

# Step 7 — README update

Create or update:

README.md

Include:

- Project goal
- Folder structure
- How to run the dataset pipeline
- Required packages
- Environment variables
- Output files
- Next step: LoRA fine-tuning

Example commands:

pip install pymupdf pandas tqdm python-dotenv openai

python src/extract_pdf_text.py
python src/chunk_lectures.py
python src/generate_synthetic_dataset.py
python src/validate_dataset.py

---

# Coding requirements

- Use clean Python.
- Use pathlib.
- Use tqdm progress bars.
- Add error handling for broken PDFs or missing folders.
- Do not hardcode absolute paths.
- The root should be detected as the current working directory.
- Make scripts runnable from the project root.
- Keep the code beginner-readable.

---

# Important design direction

Do NOT make this a basic Q&A dataset only.

The fine-tuning data should teach the model HOW to behave as:
1. a tutor,
2. an examiner,
3. a critic.

RAG will provide factual knowledge later.
Fine-tuning should provide behavior, structure, and educational style.

This is the key project idea.

---

# After this step

The generated and validated datasets now feed these implemented project components:

1. LoRA fine-tuning using `Qwen/Qwen2.5-0.5B-Instruct`
2. Offline RAG with BM25 and optional Chroma semantic retrieval
3. Optional online RAG through configured search providers
4. LangGraph-style multi-agent orchestration
