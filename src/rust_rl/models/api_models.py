"""
API-based model providers (Claude, ChatGPT, Grok)
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import requests

from .base import ModelProvider


class APIModelProvider(ModelProvider):
    """Provider for API-based models"""
    
    def __init__(self, name: str, model_id: str, provider: str, config: Dict[str, Any] = None):
        super().__init__(name, model_id, config)
        self.provider = provider
        self.api_key = self._get_api_key()
        self.base_url = self._get_base_url()
        self.log_path: Optional[Path] = None
        
    def _get_api_key(self) -> str:
        """Get API key from environment variables"""
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY", 
            "xai": "XAI_API_KEY",
            "google": "GOOGLE_API_KEY"
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
            "xai": "https://api.x.ai/v1/chat/completions",
            "google": "https://generativelanguage.googleapis.com/v1beta/models"
        }
        return url_map.get(self.provider)
    
    def set_log_path(self, log_path: Path):
        """Set path for API call logging"""
        self.log_path = log_path
        # Ensure parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _log_api_call(self, prompt: str, system_prompt: str, response: str, 
                      input_tokens: int, output_tokens: int, duration: float, 
                      response_json: Dict[str, Any] = None):
        """Log API call details to JSONL file"""
        if not self.log_path:
            return
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.name,
            "model_id": self.model_id,
            "provider": self.provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "duration_seconds": duration,
            "prompt_length": len(prompt),
            "system_prompt_length": len(system_prompt) if system_prompt else 0,
            "response_length": len(response),
            "config": {
                "max_tokens": self.config.get("max_new_tokens", 1024),
                "temperature": self.config.get("temperature", 0.2),
                "top_p": self.config.get("top_p", 0.9)
            }
        }
        
        # Add provider-specific metadata
        if response_json:
            if self.provider == "anthropic" and "usage" in response_json:
                log_entry["usage"] = response_json["usage"]
            elif self.provider in ["openai", "xai"] and "usage" in response_json:
                log_entry["usage"] = response_json["usage"]
            elif self.provider == "google" and "usageMetadata" in response_json:
                log_entry["usage"] = response_json["usageMetadata"]
        
        # Write to JSONL file
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Warning: Failed to log API call: {e}")
    
    def _count_tokens_estimate(self, text: str) -> int:
        """Rough token count estimation (4 chars per token average)"""
        return len(text) // 4
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using API"""
        if self.provider == "anthropic":
            return self._generate_anthropic(prompt, system_prompt)
        elif self.provider == "openai":
            return self._generate_openai(prompt, system_prompt)
        elif self.provider == "xai":
            return self._generate_xai(prompt, system_prompt)
        elif self.provider == "google":
            return self._generate_google(prompt, system_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _generate_anthropic(self, prompt: str, system_prompt: str = None) -> str:
        """Generate using Anthropic Claude API"""
        start_time = time.time()
        
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
        response_json = response.json()
        response_text = response_json["content"][0]["text"]
        
        duration = time.time() - start_time
        
        # Extract token counts from response
        input_tokens = response_json.get("usage", {}).get("input_tokens", 
                                                        self._count_tokens_estimate(prompt + (system_prompt or "")))
        output_tokens = response_json.get("usage", {}).get("output_tokens", 
                                                         self._count_tokens_estimate(response_text))
        
        # Log the API call
        self._log_api_call(prompt, system_prompt, response_text, 
                          input_tokens, output_tokens, duration, response_json)
        
        return response_text
    
    def _generate_openai(self, prompt: str, system_prompt: str = None) -> str:
        """Generate using OpenAI API"""
        start_time = time.time()
        
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
        response_json = response.json()
        response_text = response_json["choices"][0]["message"]["content"]
        
        duration = time.time() - start_time
        
        # Extract token counts from response
        input_tokens = response_json.get("usage", {}).get("prompt_tokens", 
                                                        self._count_tokens_estimate(prompt + (system_prompt or "")))
        output_tokens = response_json.get("usage", {}).get("completion_tokens", 
                                                         self._count_tokens_estimate(response_text))
        
        # Log the API call
        self._log_api_call(prompt, system_prompt, response_text, 
                          input_tokens, output_tokens, duration, response_json)
        
        return response_text
    
    def _generate_xai(self, prompt: str, system_prompt: str = None) -> str:
        """Generate using xAI Grok API"""
        start_time = time.time()
        
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
        response_json = response.json()
        response_text = response_json["choices"][0]["message"]["content"]
        
        duration = time.time() - start_time
        
        # Extract token counts from response
        input_tokens = response_json.get("usage", {}).get("prompt_tokens", 
                                                        self._count_tokens_estimate(prompt + (system_prompt or "")))
        output_tokens = response_json.get("usage", {}).get("completion_tokens", 
                                                         self._count_tokens_estimate(response_text))
        
        # Log the API call
        self._log_api_call(prompt, system_prompt, response_text, 
                          input_tokens, output_tokens, duration, response_json)
        
        return response_text
    
    def _generate_google(self, prompt: str, system_prompt: str = None) -> str:
        """Generate using Google Gemini API"""
        start_time = time.time()
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Build the URL with API key as a query parameter
        url = f"{self.base_url}/{self.model_id}:generateContent?key={self.api_key}"
        
        # Build content parts
        parts = []
        if system_prompt:
            parts.append({"text": system_prompt})
        parts.append({"text": prompt})
        
        data = {
            "contents": [
                {
                    "parts": parts
                }
            ],
            "generationConfig": {
                "maxOutputTokens": self.config.get("max_new_tokens", 1024),
                "temperature": self.config.get("temperature", 0.2),
                "topP": self.config.get("top_p", 0.9)
            }
        }
        
        response = self._make_request(url, headers, data)
        response_json = response.json()
        
        duration = time.time() - start_time
        
        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            candidate = response_json["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                response_text = candidate["content"]["parts"][0]["text"]
                
                # Extract token counts from response
                input_tokens = response_json.get("usageMetadata", {}).get("promptTokenCount", 
                                                               self._count_tokens_estimate(prompt + (system_prompt or "")))
                output_tokens = response_json.get("usageMetadata", {}).get("candidatesTokenCount", 
                                                                         self._count_tokens_estimate(response_text))
                
                # Log the API call
                self._log_api_call(prompt, system_prompt, response_text, 
                                  input_tokens, output_tokens, duration, response_json)
                
                return response_text
        
        raise ValueError("Invalid response format from Google Gemini API")
    
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