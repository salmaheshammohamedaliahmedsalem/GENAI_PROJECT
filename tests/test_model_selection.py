from src.agents.adaptation_agent import StudentAdaptationAgent
from src.agents.tutor_agent import TutorAgent
from src.llm.client import ChatClient
from src.llm import model_registry
from src.llm.model_registry import ChatModelOption, get_recommended_chat_model_id, list_chat_model_options
from src.rag.hybrid_retriever import HybridRetriever
from src.schemas import DocumentChunk, RetrievedChunk


def test_model_registry_discovers_finetuned_adapters():
    options = list_chat_model_options(include_unavailable=True)
    adapter_ids = [option.id for option in options if option.kind == "lora_adapter"]

    assert any(option.is_finetuned for option in options)
    assert any(option.kind == "base_model" and not option.is_finetuned for option in options)
    assert any("qwen_0_5b_lora_adapter_salma" in option_id for option_id in adapter_ids)
    assert any(option_id.endswith("_fatma") for option_id in adapter_ids)
    assert any(option_id.endswith("_salma") for option_id in adapter_ids)
    assert any(option_id.endswith("_khadija") for option_id in adapter_ids)
    assert not any("checkpoint" in option_id for option_id in adapter_ids)
    assert get_recommended_chat_model_id() in {option.id for option in list_chat_model_options(include_unavailable=False)}


def test_chat_client_routes_to_base_model(monkeypatch):
    captured = {}

    def fake_resolve(model_selection, prefer_finetuned=False):
        return ChatModelOption(
            id="base::Qwen/Qwen2.5-0.5B-Instruct",
            label="Base model only: Qwen/Qwen2.5-0.5B-Instruct",
            kind="base_model",
            available=True,
            status="test",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
        )

    def fake_base_generate(messages, max_new_tokens=384, base_model_id=None, local_files_only=True):
        captured["messages"] = messages
        captured["base_model_id"] = base_model_id
        return "base model answer"

    monkeypatch.setattr("src.llm.client.resolve_chat_model_option", fake_resolve)
    monkeypatch.setattr("src.finetuning.inference_lora.generate_with_base_messages", fake_base_generate)

    answer = ChatClient().generate(
        [{"role": "user", "content": "Use RAG context."}],
        model_selection="base::Qwen/Qwen2.5-0.5B-Instruct",
    )

    assert answer == "base model answer"
    assert captured["base_model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert captured["messages"][0]["content"] == "Use RAG context."


def test_model_registry_lists_groq_when_key_is_configured(monkeypatch):
    monkeypatch.setattr(model_registry, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(model_registry, "GROQ_MODEL", "llama-3.1-8b-instant")

    options = model_registry.list_chat_model_options(include_unavailable=True)

    groq_options = [option for option in options if option.kind == "groq"]
    assert groq_options
    assert groq_options[0].id == "groq::llama-3.1-8b-instant"
    assert groq_options[0].available is True


def test_lora_adapter_is_unavailable_when_base_model_cache_is_missing(monkeypatch):
    monkeypatch.setattr(model_registry, "_lora_dependency_status", lambda: (True, "LoRA deps ready"))
    monkeypatch.setattr(
        model_registry,
        "_base_model_cache_status",
        lambda base_model, require_tokenizer=True: (False, f"Base model {base_model} is not cached locally"),
    )

    options = model_registry.list_chat_model_options(include_unavailable=True)
    khadija = next(option for option in options if option.id.endswith("_khadija"))

    assert khadija.available is False
    assert "not cached locally" in khadija.status


def test_chat_client_routes_to_groq_model(monkeypatch):
    captured = {}

    def fake_resolve(model_selection, prefer_finetuned=False):
        return ChatModelOption(
            id="groq::llama-3.1-8b-instant",
            label="Groq hosted model: llama-3.1-8b-instant",
            kind="groq",
            available=True,
            status="test",
            base_model="llama-3.1-8b-instant",
        )

    class FakeMessage:
        content = "groq answer"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeGroqClient:
        chat = FakeChat()

    monkeypatch.setattr("src.llm.client.resolve_chat_model_option", fake_resolve)

    client = ChatClient()
    client.groq_client = FakeGroqClient()
    answer = client.generate(
        [{"role": "user", "content": "Use RAG context with Groq."}],
        model_selection="groq::llama-3.1-8b-instant",
    )

    assert answer == "groq answer"
    assert captured["model"] == "llama-3.1-8b-instant"
    assert captured["messages"][0]["content"] == "Use RAG context with Groq."


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
        model_selection="lora::outputs/finetune/qwen_0_5b_lora_adapter_salma",
    )

    assert answer.startswith("grounded answer")
    assert "Sources used:" in answer
    prompt = captured["messages"][-1]["content"]
    assert "RAG means Retrieval-Augmented Generation" in prompt
    assert "Retrieved RAG context sent to the model" in prompt
    assert "RAG passes retrieved lecture evidence into the model prompt." in prompt
    assert "[Source: lecture.pdf, page 7, chunk rag-1]" in prompt
    assert captured["model_selection"] == "lora::outputs/finetune/qwen_0_5b_lora_adapter_salma"


def test_rag_explanation_retrieval_prioritizes_architecture_context():
    retrieved = HybridRetriever().retrieve(
        "Explain RAG from our course lectures and show retrieved evidence.",
        mode="offline_only",
    )
    top_ids = [item.chunk.chunk_id for item in retrieved[:4]]

    assert "lecture_7_p10_c1" in top_ids
    assert "lecture_7_p11_c1" in top_ids or "lecture_7_p16_c1" in top_ids
