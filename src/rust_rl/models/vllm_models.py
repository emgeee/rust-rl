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
        # Use vLLM URL from config - should always be provided by inference runner
        if config and "vllm_url" in config:
            self.base_url = config["vllm_url"]
        else:
            # Fallback for backwards compatibility
            default_host = config.get("vllm_host", "localhost") if config else "localhost"
            default_port = config.get("vllm_port", 8000) if config else 8000
            self.base_url = f"http://{default_host}:{default_port}"
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
            response_data = response.json()
            
            if "choices" not in response_data or not response_data["choices"]:
                raise RuntimeError(f"vLLM server returned empty response")
                
            return response_data["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Cannot connect to vLLM server at {self.base_url}: Connection refused. Make sure the server is running.")
        except requests.exceptions.Timeout as e:
            raise RuntimeError(f"vLLM request timed out after 300 seconds")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                raise RuntimeError(f"Model '{self.model_id}' not found on vLLM server. Check if the correct model is loaded.")
            else:
                raise RuntimeError(f"vLLM server returned HTTP {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"vLLM request failed: {e}")
        except KeyError as e:
            raise RuntimeError(f"Unexpected vLLM response format: missing {e}")
        except Exception as e:
            raise RuntimeError(f"vLLM generation failed: {e}")
    
    
    def is_available(self) -> bool:
        """Check if vLLM server is running and reachable"""
        try:
            health_url = f"{self.base_url}/health"
            response = requests.get(health_url, timeout=5)
            if response.status_code != 200:
                print(f"❌ vLLM server health check failed: HTTP {response.status_code} at {health_url}")
                return False
                
            # Also check if our specific model is available
            models_url = f"{self.base_url}/v1/models"
            models_response = requests.get(models_url, timeout=5)
            if models_response.status_code == 200:
                available_models = [m["id"] for m in models_response.json().get("data", [])]
                if self.model_id not in available_models:
                    print(f"❌ Model '{self.model_id}' not found on vLLM server. Available models: {', '.join(available_models)}")
                    return False
            else:
                print(f"⚠️  Could not verify model availability: HTTP {models_response.status_code}")
                
            return True
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Cannot connect to vLLM server at {self.base_url}: Connection refused")
            print(f"   Make sure the vLLM server is running and accessible")
            return False
        except requests.exceptions.Timeout as e:
            print(f"❌ vLLM server at {self.base_url} timed out")
            return False
        except requests.exceptions.RequestException as e:
            print(f"❌ vLLM server request failed: {e}")
            return False
        except Exception as e:
            print(f"❌ vLLM server availability check failed: {e}")
            return False