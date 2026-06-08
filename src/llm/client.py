import json
from src.config import OPENAI_API_KEY, CHAT_MODEL, USE_LOCAL_LLM
from src.llm.local_llm import LocalRuleBasedLLM
from src.llm.model_registry import resolve_chat_model_option

class ChatClient:
    def __init__(self):
        self.local_llm = LocalRuleBasedLLM()
        self.use_local = USE_LOCAL_LLM or not OPENAI_API_KEY
        self.client = None
        if not self.use_local:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=OPENAI_API_KEY)
            except Exception:
                self.use_local = True

    def backend_kind(self, model_selection: str | None = None) -> str:
        if not model_selection:
            return "local_rule_based" if self.use_local or self.client is None else "openai"
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
            if option.kind == "openai":
                if self.client is None:
                    raise RuntimeError("OpenAI model selected but OPENAI_API_KEY is not configured.")
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
                    return json.loads(raw[start:end+1])
                except Exception:
                    pass
            return {"text": raw}
