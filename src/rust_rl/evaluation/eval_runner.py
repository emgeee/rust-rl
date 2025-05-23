"""
Multi-model evaluation runner
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from ..reward_functions.utils import RustTool
from .evaluator import evaluate_solutions, print_evaluation_summary
from .config import UnifiedConfig


class EvaluationRunner:
    """Runs evaluation for multiple models' predictions"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.tools = [RustTool(tool_name) for tool_name in config.evaluation_tools]
    
    def run_evaluation_for_model(self, model_name: str, force_rerun: bool = False) -> bool:
        """
        Run evaluation for a single model's predictions
        
        Args:
            model_name: Name of the model
            force_rerun: Whether to force re-running even if results exist
            
        Returns:
            True if successful, False otherwise
        """
        output_dir = self.config.get_output_dir(model_name)
        predictions_path = output_dir / "predictions.parquet"
        results_path = output_dir / "results.parquet"
        
        # Check if predictions exist
        if not predictions_path.exists():
            print(f"Predictions not found for {model_name}: {predictions_path}")
            return False
        
        # Check if results already exist
        if results_path.exists() and not force_rerun:
            print(f"Results already exist for {model_name}: {results_path}")
            # Load and display statistics for existing results
            try:
                results_df = pd.read_parquet(results_path)
                print_evaluation_summary(results_df, self.tools)
            except Exception as e:
                print(f"⚠️  Could not load existing results for summary: {e}")
            return True
        
        try:
            print(f"Running evaluation for {model_name}")
            
            # Load predictions
            predictions_df = pd.read_parquet(predictions_path)
            print(f"Loaded {len(predictions_df)} predictions for {model_name}")
            
            # Run evaluation using existing evaluator
            results_df = evaluate_solutions(
                df=predictions_df,
                tools=self.tools,
                output_file=str(results_path),
                progress_bar=tqdm,
                max_rows=-1  # Evaluate all rows
            )
            
            print(f"Evaluation completed for {model_name}: {len(results_df)} results saved")
            return True
            
        except Exception as e:
            print(f"Failed to run evaluation for {model_name}: {e}")
            return False
    
    def run_evaluation_for_all_models(self, force_rerun: bool = False, selected_models: List[str] = None) -> Dict[str, bool]:
        """
        Run evaluation for all models that have predictions
        
        Args:
            force_rerun: Whether to force re-running even if results exist
            selected_models: List of model names to evaluate (None for all)
            
        Returns:
            Dictionary mapping model names to success status
        """
        # Get all models that have predictions
        available_models = []
        for model_config in self.config.get_all_models():
            predictions_path = self.config.get_output_dir(model_config.name) / "predictions.parquet"
            if predictions_path.exists():
                available_models.append(model_config.name)
        
        if selected_models:
            available_models = [m for m in available_models if m in selected_models]
        
        if not available_models:
            print("No models with predictions found")
            return {}
        
        results = {}
        
        for model_name in available_models:
            print(f"\n{'='*60}")
            print(f"Running evaluation for: {model_name}")
            print(f"{'='*60}")
            
            success = self.run_evaluation_for_model(model_name, force_rerun)
            results[model_name] = success
            
            if success:
                print(f"✅ {model_name}: SUCCESS")
            else:
                print(f"❌ {model_name}: FAILED")
        
        return results
    
    def get_completed_evaluations(self) -> List[str]:
        """Get list of models that have completed evaluation"""
        completed = []
        for model_config in self.config.get_all_models():
            results_path = self.config.get_output_dir(model_config.name) / "results.parquet"
            if results_path.exists():
                completed.append(model_config.name)
        return completed
    
    def get_model_results(self, model_name: str) -> pd.DataFrame:
        """Get evaluation results for a specific model"""
        results_path = self.config.get_output_dir(model_name) / "results.parquet"
        if not results_path.exists():
            raise FileNotFoundError(f"Results not found for {model_name}: {results_path}")
        return pd.read_parquet(results_path)
    
    def get_all_results(self) -> Dict[str, pd.DataFrame]:
        """Get evaluation results for all completed models"""
        results = {}
        for model_name in self.get_completed_evaluations():
            try:
                results[model_name] = self.get_model_results(model_name)
            except Exception as e:
                print(f"Failed to load results for {model_name}: {e}")
        return results