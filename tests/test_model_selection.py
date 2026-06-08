from src.agents.adaptation_agent import StudentAdaptationAgent
from src.agents.tutor_agent import TutorAgent
from src.llm.client import ChatClient
from src.llm.model_registry import get_recommended_chat_model_id, list_chat_model_options
from src.schemas import DocumentChunk, RetrievedChunk


def test_model_registry_discovers_finetuned_adapters():
    options = list_chat_model_options(include_unavailable=True)

    assert any(option.is_finetuned for option in options)
    assert any("qwen_0_5b_lora_adapter" in option.id for option in options)
    assert get_recommended_chat_model_id() in {option.id for option in list_chat_model_options(include_unavailable=False)}


def test_tutor_prompt_sends_retrieved_context_to_selected_llm(monkeypatch):
    captured = {}

    def fake_backend_kind(self, model_selection=None):
        return "lora_adapter"

    def fake_generate(self, messages, temperature=0.2, tools=None, model_selection=None, max_new_tokens=384):
        captured["messages"] = messages
        captured["model_selection"] = model_selection
        return "grounded answer"

    monkeypatch.setattr(ChatClient, "backend_kind", fake_backend_kind)
    monkeypatch.setattr(ChatClient, "generate", fake_generate)

    chunk = DocumentChunk(
        chunk_id="rag-1",
        text="RAG passes retrieved lecture evidence into the model prompt.",
        source="lecture.pdf",
        source_type="course_pdf",
        page=7,
    )
    retrieved = [RetrievedChunk(chunk=chunk, final_score=1.0)]

    answer = TutorAgent().answer(
        "Explain RAG",
        retrieved,
        "offline_only",
        student_profile=StudentAdaptationAgent().run("advanced"),
        model_selection="lora::outputs/finetune/qwen_0_5b_lora_adapter",
    )

    assert answer.startswith("grounded answer")
    assert "Sources used:" in answer
    prompt = captured["messages"][-1]["content"]
    assert "Retrieved RAG context sent to the model" in prompt
    assert "RAG passes retrieved lecture evidence into the model prompt." in prompt
    assert "[Source: lecture.pdf, page 7, chunk rag-1]" in prompt
    assert captured["model_selection"] == "lora::outputs/finetune/qwen_0_5b_lora_adapter"
