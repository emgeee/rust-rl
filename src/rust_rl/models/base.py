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
    
    def generate_batch(self, prompts: List[str], system_prompt: str = None, batch_size: int = 10) -> List[str]:
        """
        Generate responses for multiple prompts (default implementation uses sequential calls)
        
        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt
            batch_size: Batch size for processing (ignored in default implementation)
            
        Returns:
            List of generated responses
        """
        results = []
        for prompt in prompts:
            try:
                response = self.generate(prompt, system_prompt)
                results.append(response)
            except Exception as e:
                results.append(f"ERROR: {str(e)}")
        return results
    
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