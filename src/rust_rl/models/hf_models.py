"""
HuggingFace transformers model provider for local models
"""

import torch
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import ModelProvider


class HuggingFaceModelProvider(ModelProvider):
    """Provider for local HuggingFace models using transformers"""
    
    def __init__(self, name: str, model_id: str, config: Dict[str, Any] = None):
        super().__init__(name, model_id, config)
        self.device = config.get("device", "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            print(f"Loading model: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Failed to load model {self.model_id}: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using local HuggingFace model"""
        if not self.model or not self.tokenizer:
            raise RuntimeError(f"Model {self.model_id} not loaded")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        input_text = input_text + "<|im_start|>assistant\n"
        
        # Tokenize and generate
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        generation_params = self.get_generation_params()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=generation_params["max_new_tokens"],
                temperature=generation_params["temperature"],
                top_p=generation_params["top_p"],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs[0]):], skip_special_tokens=True
        )
        return response.strip()
    
    def is_available(self) -> bool:
        """Check if model is loaded and available"""
        return self.model is not None and self.tokenizer is not None