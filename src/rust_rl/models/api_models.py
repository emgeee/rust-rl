"""
API-based model providers (Claude, ChatGPT, Grok)
"""

import os
import time
from typing import Dict, Any
import requests

from .base import ModelProvider


class APIModelProvider(ModelProvider):
    """Provider for API-based models"""
    
    def __init__(self, name: str, model_id: str, provider: str, config: Dict[str, Any] = None):
        super().__init__(name, model_id, config)
        self.provider = provider
        self.api_key = self._get_api_key()
        self.base_url = self._get_base_url()
        
    def _get_api_key(self) -> str:
        """Get API key from environment variables"""
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY", 
            "xai": "XAI_API_KEY"
        }
        key_name = key_map.get(self.provider)
        if not key_name:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        api_key = os.getenv(key_name)
        if not api_key:
            raise ValueError(f"API key not found: {key_name}")
        return api_key
    
    def _get_base_url(self) -> str:
        """Get base URL for the API provider"""
        url_map = {
            "anthropic": "https://api.anthropic.com/v1/messages",
            "openai": "https://api.openai.com/v1/chat/completions",
            "xai": "https://api.x.ai/v1/chat/completions"
        }
        return url_map.get(self.provider)
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using API"""
        if self.provider == "anthropic":
            return self._generate_anthropic(prompt, system_prompt)
        elif self.provider == "openai":
            return self._generate_openai(prompt, system_prompt)
        elif self.provider == "xai":
            return self._generate_xai(prompt, system_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _generate_anthropic(self, prompt: str, system_prompt: str = None) -> str:
        """Generate using Anthropic Claude API"""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        data = {
            "model": self.model_id,
            "max_tokens": self.config.get("max_new_tokens", 1024),
            "temperature": self.config.get("temperature", 0.2),
            "messages": messages
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        response = self._make_request(self.base_url, headers, data)
        return response.json()["content"][0]["text"]
    
    def _generate_openai(self, prompt: str, system_prompt: str = None) -> str:
        """Generate using OpenAI API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model_id,
            "max_tokens": self.config.get("max_new_tokens", 1024),
            "temperature": self.config.get("temperature", 0.2),
            "messages": messages
        }
        
        response = self._make_request(self.base_url, headers, data)
        return response.json()["choices"][0]["message"]["content"]
    
    def _generate_xai(self, prompt: str, system_prompt: str = None) -> str:
        """Generate using xAI Grok API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model_id,
            "max_tokens": self.config.get("max_new_tokens", 1024),
            "temperature": self.config.get("temperature", 0.2),
            "messages": messages
        }
        
        response = self._make_request(self.base_url, headers, data)
        return response.json()["choices"][0]["message"]["content"]
    
    def _make_request(self, url: str, headers: Dict, data: Dict, max_retries: int = 3) -> requests.Response:
        """Make API request with retry logic"""
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=120)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                wait_time = 2 ** attempt
                print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    def is_available(self) -> bool:
        """Check if API is available"""
        try:
            # Make a simple test request
            test_response = self.generate("Test", "You are a helpful assistant.")
            return len(test_response) > 0
        except Exception:
            return False