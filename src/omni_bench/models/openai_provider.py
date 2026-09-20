import os
import json
import urllib.request
import urllib.error
from typing import Optional
from .base import BaseModelProvider

class OpenAICompatibleProvider(BaseModelProvider):
    """Executes models served via any OpenAI-compatible API endpoint (vLLM, LMDeploy, OpenAI, Groq, DeepSeek)."""

    def __init__(self, model_name: str, base_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[ERROR: OpenAI/vLLM generation failed: {str(e)}]"
