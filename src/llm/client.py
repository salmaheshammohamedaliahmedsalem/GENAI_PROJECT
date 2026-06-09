import json
from src.config import CHAT_MODEL, GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, OPENAI_API_KEY, USE_LOCAL_LLM
from src.llm.local_llm import LocalRuleBasedLLM
from src.llm.model_registry import resolve_chat_model_option

class ChatClient:
    def __init__(self):
        self.local_llm = LocalRuleBasedLLM()
        self.openai_client = None
        self.groq_client = None
        if not USE_LOCAL_LLM:
            try:
                from openai import OpenAI
                if OPENAI_API_KEY:
                    self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                if GROQ_API_KEY:
                    self.groq_client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
            except Exception:
                self.openai_client = None
                self.groq_client = None
        self.client = self.openai_client
        self.use_local = USE_LOCAL_LLM or (self.openai_client is None and self.groq_client is None)

    def backend_kind(self, model_selection: str | None = None) -> str:
        if not model_selection:
            if self.use_local:
                return "local_rule_based"
            if self.groq_client is not None:
                return "groq"
            return "openai" if self.openai_client is not None else "local_rule_based"
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
            if option.kind == "base_model":
                from src.finetuning.inference_lora import generate_with_base_messages

                return generate_with_base_messages(
                    messages,
                    max_new_tokens=max_new_tokens,
                    base_model_id=option.base_model or CHAT_MODEL,
                )
            if option.kind == "openai":
                if self.openai_client is None:
                    raise RuntimeError("OpenAI model selected but OPENAI_API_KEY is not configured.")
                response = self.openai_client.chat.completions.create(
                    model=option.base_model or CHAT_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                return response.choices[0].message.content or ""
            if option.kind == "groq":
                if self.groq_client is None:
                    raise RuntimeError("Groq model selected but GROQ_API_KEY is not configured.")
                response = self.groq_client.chat.completions.create(
                    model=option.base_model or GROQ_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                return response.choices[0].message.content or ""

        if self.use_local:
            return self.local_llm.generate(messages, temperature=temperature, tools=tools)
        if self.groq_client is not None:
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            return response.choices[0].message.content or ""
        response = self.openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
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
                    return json.loads(raw[start:end+1])
                except Exception:
                    pass
            return {"text": raw}
