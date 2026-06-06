# 🚀 Adaptive Multi-Agent GenAI Tutor with Hybrid RAG

Welcome to the dataset preparation and processing pipeline for the **Adaptive Multi-Agent GenAI Tutor with Hybrid Online/Offline RAG**. 

This system is designed to ingest academic lecture PDFs and synthesize them into structured instruction-tuning datasets. These datasets teach a foundation LLM the distinct cognitive behavioral formats required to act as an educational **Tutor**, an academic **Examiner**, and a reflective **Critic**.

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
│   ├── raw_pdfs/              # Raw source lecture PDFs (LLM Lectures 1 - 9)
│   ├── extracted_text/        # Normalized page-by-page JSONL extractions
│   ├── chunks/                # Word-bounded slide/page text chunks
│   ├── finetuning/            # Raw and clean target fine-tuning datasets
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
│   └── rag_db/                # Saved Vector DB stores (for subsequent offline RAG stage)
├── notebooks/
│   └── 01_dataset_generation_pipeline.ipynb  # Interactive walkthrough notebook
├── src/
│   ├── utils.py               # Shared pathing, logging, and keyword classifiers
│   ├── extract_pdf_text.py    # PyMuPDF text extractor
│   ├── chunk_lectures.py      # Slide/page partitioner
│   ├── generate_synthetic_dataset.py # Hybrid API & Procedural dataset synthesizer
│   └── validate_dataset.py    # Formatting, schema, and deduplication validator
└── README.md                  # Project overview & running instructions
```

---

## 🛠️ Required Packages & Installation

The pipeline relies on `PyMuPDF` for PDF reading, `tqdm` for interactive command line progress, and `openai` / `python-dotenv` for API generation.

```bash
pip install pymupdf pandas tqdm python-dotenv openai
```

---

## 🔑 Environment Variables Setup

Create a `.env` file at the root of the project to set up LLM generation API keys:

```ini
# To use OpenAI models (GPT-4o-mini) for synthetic generation
OPENAI_API_KEY=your_openai_api_key_here

# To use Groq models (Llama-3.1-8b-instant) for synthetic generation
GROQ_API_KEY=your_groq_api_key_here
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

## 🔮 Next Step: LoRA Fine-Tuning

Once the datasets are prepared and validated, we will move forward to:
1. **LoRA SFT Alignment**: Fine-tune **Qwen2.5-1.5B-Instruct** using the `combined_dataset_clean.jsonl` to teach the model to switch personas flawlessly based on the system instructions.
2. **Offline RAG Integration**: Set up a **ChromaDB** or **FAISS** vector store using the generated chunks to support live context injections.
3. **Multi-Agent Orchestration**: Connect the Tutor, Examiner, and Critic agents using **LangGraph** to construct the complete adaptive online/offline student learning portal.
