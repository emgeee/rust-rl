"""
Base model provider interface
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class ModelProvider(ABC):
    """Base class for model providers"""
    
    def __init__(self, name: str, model_id: str, config: Dict[str, Any] = None):
        self.name = name
        self.model_id = model_id
        self.config = config or {}
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate a response for the given prompt
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model provider is available and ready to use"""
        pass
    
    def get_generation_params(self) -> Dict[str, Any]:
        """Get generation parameters for this model"""
        return {
            "max_new_tokens": self.config.get("max_new_tokens", 1024),
            "temperature": self.config.get("temperature", 0.2),
            "top_p": self.config.get("top_p", 0.9),
        }