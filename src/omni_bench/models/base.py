import abc
from typing import Optional

class BaseModelProvider(abc.ABC):
    """Abstract interface for all model backends in OmniAgent-Bench."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abc.abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """Generates a text completion from the model."""
        pass

    def cleanup(self):
        """Releases memory or connections after benchmark completion (essential for multi-model runs)."""
        pass
