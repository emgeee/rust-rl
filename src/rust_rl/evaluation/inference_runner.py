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
            
            # Add dynamic server reference if available
            if '_dynamic_server' in generation_params:
                config_dict["_dynamic_server"] = generation_params['_dynamic_server']
            
            return VLLMModelProvider(
                name=model_config.name,
                model_id=model_config.model_id,
                config=config_dict
            )


class InferenceRunner:
    """Runs inference for multiple models on the evaluation dataset"""
    
    def __init__(self, config: UnifiedConfig, dynamic_server=None):
        self.config = config
        self.dynamic_server = dynamic_server
        self.dataset = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load the evaluation dataset"""
        if not os.path.exists(self.config.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset_path}")
        
        self.dataset = pd.read_parquet(self.config.dataset_path)
        print(f"Loaded dataset with {len(self.dataset)} samples")
    
    def run_inference_for_model(self, model_config: ModelConfig, force_rerun: bool = False) -> bool:
        """
        Run inference for a single model
        
        Args:
            model_config: Configuration for the model
            force_rerun: Whether to force re-running even if results exist
            
        Returns:
            True if successful, False otherwise
        """
        output_dir = self.config.get_output_dir(model_config.name)
        predictions_path = output_dir / "predictions.parquet"
        
        # Check if results already exist
        if predictions_path.exists() and not force_rerun:
            print(f"Predictions already exist for {model_config.name}: {predictions_path}")
            return True
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create model provider
            print(f"Creating model provider for {model_config.name}")
            generation_params = self.config.generation_params.copy()
            generation_params['_unified_config'] = self.config  # Pass config for vLLM URL
            if self.dynamic_server:
                generation_params['_dynamic_server'] = self.dynamic_server
            model_provider = ModelFactory.create_model(model_config, generation_params)
            
            # Set up API call logging for API models
            if hasattr(model_provider, 'set_log_path'):
                api_log_path = output_dir / "api_calls.jsonl"
                model_provider.set_log_path(api_log_path)
                print(f"API call logging enabled: {api_log_path}")
            
            # Check if model is available
            if not model_provider.is_available():
                print(f"Model {model_config.name} is not available")
                return False
            
            print(f"Running inference for {model_config.name}")
            results = []
            
            # Progress bar
            with tqdm(total=len(self.dataset), desc=f"Inference {model_config.name}") as pbar:
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
                        print(f"Error generating response for {model_config.name}, task {row['task_id']}: {e}")
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
            print(f"Inference completed for {model_config.name}: {len(results)} predictions saved")
            return True
            
        except Exception as e:
            print(f"Failed to run inference for {model_config.name}: {e}")
            return False
    
    def _save_predictions(self, results: List[Dict], filepath: Path):
        """Save predictions to parquet file"""
        df = pd.DataFrame(results)
        df.to_parquet(filepath, index=False)
    
    def run_inference_for_all_models(self, force_rerun: bool = False, selected_models: List[str] = None) -> Dict[str, bool]:
        """
        Run inference for all configured models
        
        Args:
            force_rerun: Whether to force re-running even if results exist
            selected_models: List of model names to run (None for all)
            
        Returns:
            Dictionary mapping model names to success status
        """
        all_models = self.config.get_all_models()
        
        if selected_models:
            all_models = [m for m in all_models if m.name in selected_models]
        
        results = {}
        
        for model_config in all_models:
            print(f"\n{'='*60}")
            print(f"Running inference for: {model_config.name}")
            print(f"Model ID: {model_config.model_id}")
            print(f"{'='*60}")
            
            success = self.run_inference_for_model(model_config, force_rerun)
            results[model_config.name] = success
            
            if success:
                print(f"✅ {model_config.name}: SUCCESS")
            else:
                print(f"❌ {model_config.name}: FAILED")
        
        return results
    
    def get_completed_models(self) -> List[str]:
        """Get list of models that have completed inference"""
        completed = []
        for model_config in self.config.get_all_models():
            predictions_path = self.config.get_output_dir(model_config.name) / "predictions.parquet"
            if predictions_path.exists():
                completed.append(model_config.name)
        return completed