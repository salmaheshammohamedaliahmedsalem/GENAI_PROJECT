import json
from src.config import OPENAI_API_KEY, CHAT_MODEL, USE_LOCAL_LLM
from src.llm.local_llm import LocalRuleBasedLLM

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

    def generate(self, messages: list[dict], temperature: float = 0.2, tools=None) -> str:
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
