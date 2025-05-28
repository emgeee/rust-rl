"""
Multi-model inference runner
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from ..models import ModelProvider, APIModelProvider, VLLMModelProvider
from ..prompts import RUST_SYSTEM_PROMPT
from .config import UnifiedConfig, ModelConfig


class ModelFactory:
    """Factory for creating model providers"""
    
    @staticmethod
    def create_model(model_config: ModelConfig, generation_params: Dict[str, Any]) -> ModelProvider:
        """Create a model provider from configuration"""
        config_dict = model_config.to_dict()
        config_dict.update(generation_params)
        
        if model_config.provider in ["anthropic", "openai", "xai", "google"]:
            return APIModelProvider(
                name=model_config.name,
                model_id=model_config.model_id,
                provider=model_config.provider,
                config=config_dict
            )
        else:
            # All other models are vLLM models (no more local HuggingFace models)
            # Add vLLM URL from generation params if available
            if '_unified_config' in generation_params:
                config_dict["vllm_url"] = generation_params['_unified_config'].get_vllm_server_url()
            else:
                config_dict["vllm_url"] = "http://localhost:8000"  # Default fallback
            
            
            return VLLMModelProvider(
                name=model_config.name,
                model_id=model_config.model_id,
                config=config_dict
            )


class InferenceRunner:
    """Runs inference for multiple models on the evaluation dataset"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.dataset = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load the evaluation dataset"""
        if not os.path.exists(self.config.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset_path}")
        
        self.dataset = pd.read_parquet(self.config.dataset_path)
        
        # Limit dataset rows if specified in config
        if self.config.eval_dataset_rows is not None:
            self.dataset = self.dataset.head(self.config.eval_dataset_rows)
    
    def run_inference_for_model(self, model_config: ModelConfig, force_rerun: bool = False, batch_size: int = None) -> bool:
        """
        Run inference for a single model
        
        Args:
            model_config: Configuration for the model
            force_rerun: Whether to force re-running even if results exist
            batch_size: Number of requests to process in parallel (None for auto-detection)
            
        Returns:
            True if successful, False otherwise
        """
        output_dir = self.config.get_output_dir(model_config.name)
        predictions_path = output_dir / "predictions.parquet"
        
        # Check if results already exist
        if predictions_path.exists() and not force_rerun:
            return True
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create model provider
            generation_params = self.config.generation_params.copy()
            generation_params['_unified_config'] = self.config  # Pass config for vLLM URL
            model_provider = ModelFactory.create_model(model_config, generation_params)
            
            # Set up API call logging for API models
            if hasattr(model_provider, 'set_log_path'):
                api_log_path = output_dir / "api_calls.jsonl"
                model_provider.set_log_path(api_log_path)
            
            # Check if model is available
            print(f"🔍 Checking availability for {model_config.name}...")
            if not model_provider.is_available():
                print(f"⚠️  Skipping {model_config.name} - model unavailable")
                return False
            print(f"✅ {model_config.name} is available")
            
            # Determine batch processing strategy
            use_batch_processing = self._should_use_batch_processing(model_config, batch_size)
            
            if use_batch_processing:
                return self._run_batch_inference(model_provider, model_config, predictions_path, batch_size)
            else:
                return self._run_sequential_inference(model_provider, model_config, predictions_path)
            
        except Exception as e:
            print(f"❌ Failed to run inference for {model_config.name}: {str(e)}")
            return False
    
    def _should_use_batch_processing(self, model_config: ModelConfig, batch_size: int = None) -> bool:
        """Determine if batch processing should be used for this model"""
        # Only use batch processing for API models
        api_providers = ["anthropic", "openai", "xai", "google"]
        return model_config.provider in api_providers
    
    def _run_batch_inference(self, model_provider, model_config: ModelConfig, predictions_path: Path, batch_size: int = None) -> bool:
        """Run inference using batch processing for API models"""
        if batch_size is None:
            # Determine optimal batch size based on provider
            batch_size_map = {
                "anthropic": 5,  # Conservative for Claude
                "openai": 10,   # OpenAI can handle more concurrent requests
                "xai": 8,       # Grok rate limits
                "google": 15    # Gemini is generally fast
            }
            batch_size = batch_size_map.get(model_config.provider, 5)
        
        print(f"🚀 Using batch processing with batch size {batch_size}")
        
        # Prepare all prompts
        prompts = [row['rust_prompt'] for _, row in self.dataset.iterrows()]
        
        # Generate all responses in batches
        with tqdm(total=len(prompts), desc=f"Batch Inference {model_config.name}") as pbar:
            all_responses = model_provider.generate_batch(
                prompts=prompts,
                system_prompt=RUST_SYSTEM_PROMPT,
                batch_size=batch_size
            )
            pbar.update(len(prompts))
        
        # Prepare results
        results = []
        for i, (_, row) in enumerate(self.dataset.iterrows()):
            response = all_responses[i] if i < len(all_responses) else "ERROR: No response"
            result = {
                "task_id": row["task_id"],
                "prompt": row["rust_prompt"],
                "test_list": row["rust_test_list"],
                "response": response,
                "model_name": model_config.name
            }
            results.append(result)
        
        # Save results
        self._save_predictions(results, predictions_path)
        return True
    
    def _run_sequential_inference(self, model_provider, model_config: ModelConfig, predictions_path: Path) -> bool:
        """Run inference using sequential processing (fallback for non-API models)"""
        print(f"🔄 Using sequential processing")
        
        results = []
        
        # Progress bar
        with tqdm(total=len(self.dataset), desc=f"Sequential Inference {model_config.name}") as pbar:
            for index, row in self.dataset.iterrows():
                try:
                    # Generate response
                    response = model_provider.generate(
                        prompt=row['rust_prompt'],
                        system_prompt=RUST_SYSTEM_PROMPT
                    )
                    
                    # Store result
                    result = {
                        "task_id": row["task_id"],
                        "prompt": row["rust_prompt"],
                        "test_list": row["rust_test_list"],
                        "response": response,
                        "model_name": model_config.name
                    }
                    results.append(result)
                    
                    # Save periodically
                    if len(results) % self.config.save_every == 0:
                        self._save_predictions(results, predictions_path)
                    
                    pbar.update(1)
                    
                except Exception as e:
                    # Add failed result
                    result = {
                        "task_id": row["task_id"],
                        "prompt": row["rust_prompt"],
                        "test_list": row["rust_test_list"],
                        "response": f"ERROR: {str(e)}",
                        "model_name": model_config.name
                    }
                    results.append(result)
                    pbar.update(1)
        
        # Save final results
        self._save_predictions(results, predictions_path)
        return True
    
    def _save_predictions(self, results: List[Dict], filepath: Path):
        """Save predictions to parquet file"""
        df = pd.DataFrame(results)
        df.to_parquet(filepath, index=False)
    
    def run_inference_for_all_models(self, force_rerun: bool = False, selected_models: List[str] = None, batch_size: int = None) -> Dict[str, bool]:
        """
        Run inference for all configured models
        
        Args:
            force_rerun: Whether to force re-running even if results exist
            selected_models: List of model names to run (None for all)
            batch_size: Number of requests to process in parallel for API models
            
        Returns:
            Dictionary mapping model names to success status
        """
        all_models = self.config.get_all_models()
        
        if selected_models:
            all_models = [m for m in all_models if m.name in selected_models]
        
        results = {}
        
        for model_config in all_models:
            success = self.run_inference_for_model(model_config, force_rerun, batch_size)
            results[model_config.name] = success
        
        return results
    
    def get_completed_models(self) -> List[str]:
        """Get list of models that have completed inference"""
        completed = []
        for model_config in self.config.get_all_models():
            predictions_path = self.config.get_output_dir(model_config.name) / "predictions.parquet"
            if predictions_path.exists():
                completed.append(model_config.name)
        return completed