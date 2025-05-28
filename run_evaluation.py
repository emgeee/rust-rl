#!/usr/bin/env python3
"""
Simple Evaluation Workflow Script

Runs the complete evaluation pipeline: inference -> evaluation -> visualization.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rust_rl.evaluation.config import UnifiedConfig
from rust_rl.evaluation.orchestration import EvaluationOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation workflow for Rust code generation models"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="multi_model_eval_config.yaml",
        help="Path to configuration file"
    )
    
    # Pipeline stages
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run complete pipeline (default)"
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run inference only"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true", 
        help="Run evaluation only"
    )
    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="Run visualization only"
    )
    
    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model names to process"
    )
    
    # Options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if results exist"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for parallel API requests (default: auto-detect per provider)"
    )
    
    # Server configuration
    parser.add_argument(
        "--vllm-host",
        type=str,
        help="vLLM server host (overrides config and VLLM_SERVER_HOST env var)"
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        help="vLLM server port (overrides config and VLLM_SERVER_PORT env var)"
    )
    
    args = parser.parse_args()
    
    # Default to --all if no stage specified
    if not any([args.all, args.inference_only, args.eval_only, args.viz_only]):
        args.all = True
    
    # Check config exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Set environment variables from command line args if provided
    if args.vllm_host:
        os.environ["VLLM_SERVER_HOST"] = args.vllm_host
    if args.vllm_port:
        os.environ["VLLM_SERVER_PORT"] = str(args.vllm_port)
    
    try:
        config = UnifiedConfig.from_yaml(args.config)
        # Enable silent inference mode for cleaner progress bars
        orchestrator = EvaluationOrchestrator(config, silent_inference=True)
        
        print("🦀 Rust Code Evaluation Pipeline")
        print("=" * 50)
        
        if args.dry_run:
            print("🔍 DRY RUN - No actual execution")
            orchestrator.validate_config(selected_models=args.models)
            return
        
        # Run requested stages
        if args.all:
            orchestrator.run_full_pipeline(
                selected_models=args.models,
                force_rerun=args.force,
                batch_size=args.batch_size
            )
        elif args.inference_only:
            orchestrator.run_inference_stage(
                selected_models=args.models,
                force_rerun=args.force,
                batch_size=args.batch_size
            )
        elif args.eval_only:
            orchestrator.run_evaluation_stage(
                selected_models=args.models,
                force_rerun=args.force
            )
        elif args.viz_only:
            orchestrator.run_visualization_stage(
                selected_models=args.models
            )
        
        print("🎉 Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()