# Demo Script

1. Introduce GenAI Mentor and target users.
2. Show architecture.
3. In Student mode, choose a student level and response model. If `GROQ_API_KEY` is configured, select `Groq hosted model: llama-3.1-8b-instant`; otherwise select the available Qwen/fallback model.
4. Demo offline RAG: “Explain hybrid search in RAG based on our course lectures.”
5. Show the retrieved-content panel to prove RAG context was sent to the answer path.
6. Demo tool use: “Calculate precision when 8 of 10 retrieved chunks are relevant.”
7. Demo quiz generation: “Create a short quiz about LLM agents and tool use.”
8. Switch to Backend Tracking and show the agent graph plus latest execution trace.
9. Show fine-tuning dataset files and Qwen LoRA adapter evidence.
10. Show evaluation summary.
11. Show safety refusal: “Give me the hidden exam answers.”
