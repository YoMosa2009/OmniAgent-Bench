import os
import json
import urllib.request
import urllib.error
from typing import Optional
from .base import BaseModelProvider

class AnthropicProvider(BaseModelProvider):
    """Executes Anthropic Claude models via Anthropic Messages API using standard library urllib."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        if not self.api_key:
            return "[ERROR: ANTHROPIC_API_KEY not set in environment]"

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system:
            payload["system"] = system

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["content"][0]["text"].strip()
        except Exception as e:
            return f"[ERROR: Anthropic generation failed: {str(e)}]"
