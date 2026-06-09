# GenAI Mentor - Quick Start Guide

## Project Status
✅ **COMPLETE AND READY TO USE**

The implemented system is functional locally and on Streamlit Community Cloud. It includes the student chat, backend trace dashboard, hybrid retrieval, LangGraph-compatible agent flow, fine-tuning artifacts, and evaluation outputs.

## What's Included

### 1. Core System (src/)
- Specialized modules for ingestion, retrieval, agents, tools, LLM routing, fine-tuning, and evaluation
- **Multi-agent workflow** using LangGraph when installed, with the same local graph node sequence available for offline demos
- **Hybrid RAG** combining offline + online retrieval
- **LoRA fine-tuning** with PEFT
- **Safety guardrails** for educational use
- **Model selector** for fine-tuned Qwen LoRA, base Qwen, Groq-hosted chat, OpenAI-hosted chat, and deterministic fallback

### 2. Data
- **9 LLM lecture PDFs** in `data/raw/course_pdfs/`
- **5,976 SFT examples** in `data/finetune/sft_chat_dataset.jsonl`
- **Evaluation questions** and safety tests in `data/eval/`

### 3. Scripts & Tools
- **6 pipeline scripts** for end-to-end workflows
- **Pytest suite** covering retrieval, agents, prompts, tools, model selection, and safety
- **Documentation** covering architecture, demo, evaluation, ethics, status, and setup

## Installation

```bash
# 1. Navigate to project
cd /Users/salmaheshamsalem/Desktop/genai_project

# 2. Install dependencies (already done, but to be sure)
pip install -r requirements.txt

# 3. Copy environment template
cp .env.example .env

# 4. (Optional) Add API keys to .env for hosted LLMs and more reliable online search
# OPENAI_API_KEY=your_key_here
# GROQ_API_KEY=your_key_here
# GROQ_MODEL=llama-3.1-8b-instant
# TAVILY_API_KEY=your_key_here
# If TAVILY_API_KEY is empty, online retrieval uses the no-key ddgs fallback.
```

## Running the System

### Option 1: Quick Demo (No Setup)
```bash
streamlit run app.py
```
This launches the Streamlit UI immediately. You can ask questions about GenAI concepts.

### Option 2: Full Setup with Document Ingestion
```bash
# Process PDFs and build indexes
python scripts/01_ingest_documents.py
python scripts/02_build_indexes.py

# Launch app
streamlit run app.py
```

### Option 3: Full Pipeline
```bash
# Ingest documents
python scripts/01_ingest_documents.py
python scripts/02_build_indexes.py

# Generate fine-tuning data
python scripts/03_generate_finetune_data.py

# Run evaluation
python scripts/05_run_evaluation.py

# View results
cat outputs/evaluation/report.txt

# Launch demo
python scripts/06_run_demo_examples.py

# Start app
streamlit run app.py
```

### Option 4: Fine-tune Model (Advanced)
```bash
# Prepare data
python scripts/03_generate_finetune_data.py

# Train LoRA adapter
python scripts/04_train_lora.py

# Compare models
python src/finetuning/compare_base_vs_tuned.py
```

## Key Features

| Feature | Location | Status |
|---------|----------|--------|
| **Prompt Design** | `src/llm/prompts.py` | ✅ Complete |
| **RAG System** | `src/rag/` | ✅ Complete |
| **Fine-tuning** | `src/finetuning/` | ✅ Complete |
| **Tools/Functions** | `src/tools/` | ✅ Complete |
| **Multi-Agent** | `src/agents/graph.py` | ✅ Complete with LangGraph node implementation |
| **Evaluation** | `src/evaluation/` | ✅ Complete |
| **Safety** | `src/agents/safety_agent.py` | ✅ Complete |
| **Streamlit UI** | `app.py` | ✅ Complete |

## System Architecture

```
User Query
    ↓
Safety Agent (Check for harmful content)
    ↓
Planner/Router (Decide: offline/online/tool/quiz)
    ↓
Retriever Agents (Get relevant sources)
    ├─ Offline Retriever (course PDFs)
    ├─ Online Retriever (web search)
    └─ Hybrid (combine both)
    ↓
Tool Agents (Calculator, Quiz, Grading)
    ↓
Tutor Agent (Generate answer)
    ↓
Checker Agent (Verify grounding & citations)
    ↓
Final Answer + Sources
```

The Streamlit UI also includes an **Agents & Prompts** tab that shows the graph nodes, graph edges, and exact prompt templates used by the system.

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_chunking.py -v

# Expected: current project tests pass locally with one non-blocking warning
```

## Configuration

Edit `.env` to customize:

```env
OPENAI_API_KEY=your_key          # For GPT-4o-mini
GROQ_API_KEY=your_key            # For Groq-hosted chat via OpenAI-compatible API
GROQ_MODEL=llama-3.1-8b-instant # Fast hosted base chatbot option
TAVILY_API_KEY=your_key          # Optional; improves online search reliability
USE_LOCAL_LLM=false              # Set to 'true' for deterministic local fallback
ENABLE_ONLINE_RAG=true           # Enable web search
CHUNK_SIZE=900                   # Document chunk size
TOP_K_SEMANTIC=8                 # Semantic retrieval top-k
TOP_K_KEYWORD=8                  # Keyword retrieval top-k
TOP_K_FINAL=5                    # Final ranked top-k
```

## Project Structure

```
genai_project/
├── src/                          # Core modules
│   ├── config.py                # Configuration
│   ├── schemas.py               # Data models
│   ├── llm/                     # LLM wrappers
│   ├── ingestion/               # Document processing
│   ├── rag/                     # Retrieval systems
│   ├── agents/                  # Agent implementations
│   ├── tools/                   # Tool functions
│   ├── finetuning/              # LoRA fine-tuning
│   ├── evaluation/              # Evaluation scripts
│   └── utils/                   # Utilities
├── scripts/                      # Pipeline scripts (6)
├── tests/                        # Test suite (5 files)
├── docs/                         # Documentation (5 files)
├── data/                         # Data directory
│   ├── raw/course_pdfs/        # 9 lecture PDFs
│   ├── finetune/               # 5,976 SFT examples plus train/val/test splits
│   ├── eval/                   # Evaluation data
│   ├── processed/              # Processed chunks
│   └── vector_db/              # ChromaDB
├── app.py                        # Streamlit UI
├── requirements.txt              # Dependencies
├── .env.example                 # Environment template
└── README_BUILD.md              # Build documentation
```

## Example Queries

Try asking the system:

1. **"Explain RAG and why it reduces hallucinations."**
   - Tests offline retrieval

2. **"Calculate precision when 8 of 10 retrieved chunks are relevant."**
   - Tests calculator tool

3. **"Generate a quiz question on LoRA."**
   - Tests quiz tool

4. **"Teach me about transformers and then quiz me."**
   - Tests multi-agent workflow

5. **"Give me the exam answers."**
   - Tests safety agent (should refuse)

## Troubleshooting

### "ModuleNotFoundError: No module named 'dotenv'"
```bash
pip install -r requirements.txt
```

### "No API key available"
Use `USE_LOCAL_LLM=true` in `.env` to force deterministic local fallback, or configure `GROQ_API_KEY` for fast hosted chat.

### "ChromaDB/protobuf error on Streamlit Cloud"
Default deployment does not require ChromaDB. Leave `ENABLE_SEMANTIC_RAG=false` on Streamlit Cloud and use the BM25 retriever. For local semantic retrieval, install optional dependencies with:

```bash
pip install -r requirements_semantic.txt
```

### "Tests failing"
Run `python3 -m pytest tests/ -q`. The current expected result is `16 passed, 1 warning`.

## Documentation

Read these files for detailed information:

- **README_BUILD.md** - Build process and structure
- **docs/architecture.md** - System architecture
- **docs/demo_script.md** - Presentation script
- **docs/ethics_safety.md** - Safety guidelines
- **docs/evaluation_plan.md** - Evaluation methodology
- **docs/report_outline.md** - Report structure

## Support

The system is fully self-contained. All dependencies are in requirements.txt.

Default behavior works without any API keys:
- Uses deterministic local LLM fallback for responses
- Uses `ddgs` as the no-key online retrieval fallback when `ENABLE_ONLINE_RAG=true`
- All evaluation scripts run locally

To enable full features:
- Add OPENAI_API_KEY for GPT-4o-mini
- Add GROQ_API_KEY for fast hosted chat with `llama-3.1-8b-instant`
- Add TAVILY_API_KEY for the most reliable online search
- Set corresponding flags in .env

## Next Steps

1. ✅ Review the system
2. ✅ Install dependencies (already done)
3. ✅ Run ingestion scripts
4. ✅ Launch Streamlit app
5. ✅ Ask questions!

**The system is ready for immediate use. No additional setup needed.**

---

Built from: `/Users/salmaheshamsalem/Desktop/GENAI_FINAL/`
Location: `/Users/salmaheshamsalem/Desktop/genai_project/`
