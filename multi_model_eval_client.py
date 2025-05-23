#!/usr/bin/env python3
"""
Multi-Model Rust Code Evaluation Client

This script evaluates multiple AI models on their ability to generate Rust code.
It runs inference on a dataset, evaluates the generated code using Rust tooling,
and creates comparison visualizations. This client works with the vLLM server.
"""

import argparse
import sys
import subprocess
import time
import requests
from pathlib import Path
from typing import List, Optional

from src.rust_rl.evaluation.config import UnifiedConfig
from src.rust_rl.evaluation.inference_runner import InferenceRunner
from src.rust_rl.evaluation.eval_runner import EvaluationRunner
from src.rust_rl.evaluation.multi_model_visualize import MultiModelVisualizer


def print_banner():
    """Print a nice banner"""
    print("="*80)
    print("🦀 Multi-Model Rust Code Evaluation Client 🦀")
    print("="*80)


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")


def check_vllm_server(config: UnifiedConfig) -> bool:
    """Check if vLLM server is running"""
    try:
        response = requests.get(f"{config.get_vllm_server_url()}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def start_vllm_server(config: UnifiedConfig, model_id: str) -> bool:
    """Start vLLM server with specified model"""
    print(f"🚀 Starting vLLM server with model: {model_id}")
    
    try:
        cmd = [
            "python", "vllm_model_server.py",
            "--config", "multi_model_eval_config.yaml",
            "--model", model_id,
            "--daemon"
        ]
        
        subprocess.run(cmd, check=True)
        
        # Wait for server to be ready
        print("⏳ Waiting for server to start...")
        for _ in range(30):  # Wait up to 60 seconds
            if check_vllm_server(config):
                print("✅ vLLM server is ready!")
                return True
            time.sleep(2)
        
        print("❌ vLLM server failed to start within timeout")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start vLLM server: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model evaluation client for Rust code generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline for all models (requires vLLM server running)
  python multi_model_eval_client.py --config multi_model_eval_config.yaml --all

  # Run only inference for specific models
  python multi_model_eval_client.py --config multi_model_eval_config.yaml --inference --models "claude-3-5-sonnet-20241022" "Qwen/Qwen2.5-Coder-7B-Instruct"

  # Run only evaluation (requires existing predictions)
  python multi_model_eval_client.py --config multi_model_eval_config.yaml --evaluate

  # Run only visualization (requires existing results)
  python multi_model_eval_client.py --config multi_model_eval_config.yaml --visualize

  # Auto-start vLLM server for first vLLM model found
  python multi_model_eval_client.py --config multi_model_eval_config.yaml --all --auto-start-server

  # Force re-run everything
  python multi_model_eval_client.py --config multi_model_eval_config.yaml --all --force
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="multi_model_eval_config.yaml",
        help="Path to unified configuration YAML file"
    )
    
    # Pipeline stages
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run complete pipeline (inference + evaluation + visualization)"
    )
    parser.add_argument(
        "--inference", 
        action="store_true",
        help="Run inference stage only"
    )
    parser.add_argument(
        "--evaluate", 
        action="store_true",
        help="Run evaluation stage only"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Run visualization stage only"
    )
    
    # Model selection
    parser.add_argument(
        "--models", 
        nargs="+",
        help="Specific model IDs to process (default: all configured models)"
    )
    
    # Options
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-run even if results already exist"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be done without actually running"
    )
    parser.add_argument(
        "--auto-start-server",
        action="store_true",
        help="Automatically start vLLM server if needed"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.all, args.inference, args.evaluate, args.visualize]):
        print("❌ Error: Must specify at least one stage: --all, --inference, --evaluate, or --visualize")
        sys.exit(1)
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"❌ Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    print_banner()
    
    try:
        # Load configuration
        print(f"📋 Loading configuration from: {args.config}")
        config = UnifiedConfig.from_yaml(args.config)
        print(f"✅ Configuration loaded successfully")
        print(f"   - Dataset: {config.dataset_path}")
        print(f"   - Output directory: {config.output_base_dir}")
        print(f"   - Evaluation tools: {', '.join(config.evaluation_tools)}")
        
        # Show configured models
        all_models = config.get_all_models()
        print(f"   - Total models configured: {len(all_models)}")
        for model_type, models in [
            ("API", config.models_api),
            ("vLLM", config.models_vllm)
        ]:
            if models:
                print(f"     - {model_type}: {', '.join([m.name for m in models])}")
        
        # Show server configuration
        print(f"   - vLLM Server: {config.get_vllm_server_url()}")
        
        # Filter models if specified
        selected_models = args.models
        if selected_models:
            available_model_ids = [m.model for m in all_models]
            invalid_models = [m for m in selected_models if m not in available_model_ids]
            if invalid_models:
                print(f"❌ Error: Invalid model IDs: {', '.join(invalid_models)}")
                print(f"   Available models: {', '.join(available_model_ids)}")
                sys.exit(1)
            print(f"🎯 Selected models: {', '.join(selected_models)}")
            
            # Filter config models to only selected ones
            selected_model_names = []
            for model in all_models:
                if model.model in selected_models:
                    selected_model_names.append(model.name)
        else:
            selected_model_names = None
        
        # Check vLLM server status if we have vLLM models
        vllm_models = [m for m in all_models if not m.provider]
        if vllm_models and (args.all or args.inference):
            server_running = check_vllm_server(config)
            print(f"🖥️  vLLM Server status: {'✅ Running' if server_running else '❌ Not running'}")
            
            if not server_running and args.auto_start_server:
                # Start server with first vLLM model
                first_vllm_model = vllm_models[0]
                if not selected_models or first_vllm_model.model in selected_models:
                    success = start_vllm_server(config, first_vllm_model.model)
                    if not success:
                        print("❌ Failed to auto-start vLLM server")
                        sys.exit(1)
            elif not server_running:
                print("⚠️  Warning: vLLM server not running. vLLM models will fail.")
                print("💡 Start server manually or use --auto-start-server")
        
        if args.dry_run:
            print("\n🔍 DRY RUN MODE - No actual execution")
            print("✅ Configuration validation complete")
            return
        
        # Initialize runners
        inference_runner = InferenceRunner(config)
        eval_runner = EvaluationRunner(config)
        visualizer = MultiModelVisualizer(config)
        
        # Run inference stage
        if args.all or args.inference:
            print_section("INFERENCE STAGE")
            print("🤖 Running inference for selected models...")
            
            inference_results = inference_runner.run_inference_for_all_models(
                force_rerun=args.force,
                selected_models=selected_model_names
            )
            
            # Report inference results
            successful_models = [m for m, success in inference_results.items() if success]
            failed_models = [m for m, success in inference_results.items() if not success]
            
            print(f"\n📊 Inference Results:")
            print(f"   ✅ Successful: {len(successful_models)}")
            if successful_models:
                print(f"      {', '.join(successful_models)}")
            
            if failed_models:
                print(f"   ❌ Failed: {len(failed_models)}")
                print(f"      {', '.join(failed_models)}")
        
        # Run evaluation stage
        if args.all or args.evaluate:
            print_section("EVALUATION STAGE")
            print("🧪 Running evaluation for models with predictions...")
            
            eval_results = eval_runner.run_evaluation_for_all_models(
                force_rerun=args.force,
                selected_models=selected_model_names
            )
            
            # Report evaluation results
            successful_evals = [m for m, success in eval_results.items() if success]
            failed_evals = [m for m, success in eval_results.items() if not success]
            
            print(f"\n📊 Evaluation Results:")
            print(f"   ✅ Successful: {len(successful_evals)}")
            if successful_evals:
                print(f"      {', '.join(successful_evals)}")
            
            if failed_evals:
                print(f"   ❌ Failed: {len(failed_evals)}")
                print(f"      {', '.join(failed_evals)}")
        
        # Run visualization stage
        if args.all or args.visualize:
            print_section("VISUALIZATION STAGE")
            print("📈 Creating comparison visualizations...")
            
            # Get all completed results
            all_results = eval_runner.get_all_results()
            
            # Filter by selected models if specified
            if selected_model_names:
                all_results = {k: v for k, v in all_results.items() if k in selected_model_names}
            
            if not all_results:
                print("❌ No evaluation results found for visualization")
                print("   Make sure to run inference and evaluation first")
            else:
                print(f"📊 Found results for {len(all_results)} models: {', '.join(all_results.keys())}")
                
                chart_paths = visualizer.create_all_visualizations(all_results)
                
                print("\n📈 Visualizations created:")
                for chart_name, chart_path in chart_paths.items():
                    if chart_path:
                        print(f"   📄 {chart_name}: {chart_path}")
        
        print_section("PIPELINE COMPLETE")
        print("🎉 Multi-model evaluation pipeline completed successfully!")
        print(f"📁 Results available in: {config.output_base_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.dry_run:
            print("💡 This error was caught during dry-run validation")
        sys.exit(1)


if __name__ == "__main__":
    main()