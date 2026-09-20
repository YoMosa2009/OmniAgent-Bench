from .base import BaseModelProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAICompatibleProvider
from .anthropic_provider import AnthropicProvider
from .huggingface_provider import HuggingFaceProvider

def get_model_provider(backend: str, model_name: str, **kwargs) -> BaseModelProvider:
    backend_lower = backend.lower().strip()
    if backend_lower in ("ollama", "local"):
        return OllamaProvider(model_name, **kwargs)
    elif backend_lower in ("openai", "vllm", "opencode", "groq", "deepseek"):
        return OpenAICompatibleProvider(model_name, **kwargs)
    elif backend_lower in ("anthropic", "claude"):
        return AnthropicProvider(model_name, **kwargs)
    elif backend_lower in ("hf", "huggingface", "transformers"):
        return HuggingFaceProvider(model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Supported: ollama, openai (vLLM), anthropic, huggingface")
