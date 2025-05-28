"""
vLLM-based model provider for local HuggingFace models
"""

import os
import requests
from typing import Dict, Any

from .base import ModelProvider


class VLLMModelProvider(ModelProvider):
    """Provider for vLLM-served models with dynamic loading support"""
    
    def __init__(self, name: str, model_id: str, config: Dict[str, Any] = None):
        super().__init__(name, model_id, config)
        # Allow environment variable override for server URL
        default_host = config.get("vllm_host", "localhost")
        default_port = config.get("vllm_port", 8000)
        host = os.getenv("VLLM_SERVER_HOST", default_host)
        port = int(os.getenv("VLLM_SERVER_PORT", str(default_port)))
        self.base_url = f"http://{host}:{port}"
        self.endpoint = f"{self.base_url}/v1/chat/completions"
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using vLLM server"""
        
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
    
    
    def is_available(self) -> bool:
        """Check if vLLM server is running and reachable"""
        try:
            health_url = f"{self.base_url}/health"
            response = requests.get(health_url, timeout=5)
            if response.status_code != 200:
                raise RuntimeError(f"vLLM server health check failed: HTTP {response.status_code}")
            return True
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cannot reach vLLM server at {self.base_url}: {e}")
        except Exception as e:
            raise RuntimeError(f"vLLM server availability check failed: {e}")