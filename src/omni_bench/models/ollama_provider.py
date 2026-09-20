import json
import urllib.request
import urllib.error
from typing import Optional
from .base import BaseModelProvider

class OllamaProvider(BaseModelProvider):
    """Executes models served locally via Ollama using standard library urllib."""

    def __init__(self, model_name: str, host: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name, **kwargs)
        self.host = host.rstrip("/")

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        if system:
            payload["system"] = system

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except Exception as e:
            return f"[ERROR: Ollama generation failed: {str(e)}]"

    def cleanup(self):
        """Unload model from Ollama VRAM to free GPU for next model."""
        url = f"{self.host}/api/generate"
        payload = {"model": self.model_name, "keep_alive": 0}
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                pass
        except Exception:
            pass
