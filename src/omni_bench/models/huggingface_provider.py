import gc
from typing import Optional
from .base import BaseModelProvider

class HuggingFaceProvider(BaseModelProvider):
    """Executes models locally via Hugging Face Transformers with memory cleanup."""

    def __init__(self, model_name: str, torch_dtype: str = "bfloat16", device_map: str = "auto", **kwargs):
        super().__init__(model_name, **kwargs)
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        dtype = getattr(torch, self.torch_dtype) if hasattr(torch, self.torch_dtype) else torch.float16
        print(f"[HFProvider] Loading {self.model_name} onto GPU...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=True
        )
        self.model.eval()

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        import torch
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt_text = prompt
        else:
            prompt_text = prompt

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0.01

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        resp = self.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return resp

    def cleanup(self):
        """Unloads model and flushes CUDA memory so subsequent models do not OOM."""
        print(f"[HFProvider] Unloading {self.model_name} and freeing GPU memory...", flush=True)
        try:
            import torch
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print("[HFProvider] GPU memory successfully released.", flush=True)
        except Exception as e:
            print(f"[HFProvider] Cleanup warning: {e}", flush=True)
