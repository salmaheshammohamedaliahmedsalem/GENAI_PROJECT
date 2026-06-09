import json
from src.config import OPENAI_API_KEY, GROQ_API_KEY, GROQ_BASE_URL, CHAT_MODEL, USE_LOCAL_LLM
from src.llm.local_llm import LocalRuleBasedLLM
from src.llm.model_registry import resolve_chat_model_option


class ChatClient:
    def __init__(self):
        self.local_llm = LocalRuleBasedLLM()
        self.use_local = USE_LOCAL_LLM
        self.client = None
        self.client_kind = "none"

        if not self.use_local:
            try:
                from openai import OpenAI
                if GROQ_API_KEY:
                    self.client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
                    self.client_kind = "groq"
                elif OPENAI_API_KEY:
                    self.client = OpenAI(api_key=OPENAI_API_KEY)
                    self.client_kind = "openai"
            except Exception:
                pass

        if self.client is None:
            self.use_local = True

    def backend_kind(self, model_selection: str | None = None) -> str:
        if not model_selection:
            if self.use_local or self.client is None:
                return "local_rule_based"
            return self.client_kind
        option = resolve_chat_model_option(model_selection)
        if not option.available:
            return "unavailable"
        return option.kind

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        tools=None,
        model_selection: str | None = None,
        max_new_tokens: int = 384,
    ) -> str:
        if model_selection:
            option = resolve_chat_model_option(model_selection)
            if not option.available:
                raise RuntimeError(f"Selected model is unavailable: {option.status}")
            if option.kind == "lora_adapter":
                from src.finetuning.inference_lora import generate_with_lora_messages
                return generate_with_lora_messages(
                    messages,
                    max_new_tokens=max_new_tokens,
                    adapter_dir=option.path,
                    base_model_id=option.base_model or CHAT_MODEL,
                )
            if option.kind in {"openai", "groq"}:
                if self.client is None:
                    raise RuntimeError(f"{option.kind} model selected but API key is not configured.")
                response = self.client.chat.completions.create(
                    model=option.base_model or CHAT_MODEL,
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""

        if self.use_local or self.client is None:
            return self.local_llm.generate(messages, temperature=temperature, tools=tools)
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def generate_json(self, messages: list[dict], schema_hint=None, temperature: float = 0.0) -> dict:
        raw = self.generate(messages, temperature=temperature)
        try:
            return json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start:end + 1])
                except Exception:
                    pass
            return {"text": raw}
