"""
vLLM-based model provider for local HuggingFace models
"""

import requests
from typing import Dict, Any

from .base import ModelProvider


class VLLMModelProvider(ModelProvider):
    """Provider for vLLM-served models with dynamic loading support"""
    
    def __init__(self, name: str, model_id: str, config: Dict[str, Any] = None):
        super().__init__(name, model_id, config)
        self.base_url = config.get("vllm_url", "http://localhost:8000")
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.dynamic_server = config.get("_dynamic_server")  # Reference to dynamic server
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using vLLM server with automatic model loading"""
        # Automatically load model if using dynamic server
        if self.dynamic_server:
            if not self._ensure_model_loaded():
                raise RuntimeError(f"Failed to load model {self.model_id}")
        
        headers = {
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.config.get("max_new_tokens", 1024),
            "temperature": self.config.get("temperature", 0.2),
            "top_p": self.config.get("top_p", 0.9),
            "stream": False
        }
        
        try:
            response = requests.post(self.endpoint, headers=headers, json=data, timeout=300)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"vLLM request failed: {e}")
    
    def _ensure_model_loaded(self) -> bool:
        """Ensure the model is loaded using the dynamic server"""
        if not self.dynamic_server:
            return True  # Not using dynamic loading
        
        try:
            from ..evaluation.dynamic_model_server import ModelLoadQueue
            if hasattr(self.dynamic_server, 'request_model'):
                # Using queue-based loading
                return self.dynamic_server.request_model(self.model_id, f"VLLMModelProvider({self.name})")
            else:
                # Using direct dynamic server
                return self.dynamic_server.ensure_model_loaded(self.model_id)
        except Exception as e:
            print(f"❌ Error ensuring model {self.model_id} is loaded: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if model is available (either server is running or can be dynamically loaded)"""
        # If using dynamic server, check if model can be loaded
        if self.dynamic_server:
            available_models = self.dynamic_server.get_available_models()
            return self.model_id in available_models
        
        # Otherwise check if vLLM server is running
        try:
            health_url = f"{self.base_url}/health"
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False