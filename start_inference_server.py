#!/usr/bin/env python3
"""
Simple Inference Server Startup Script

Starts a vLLM inference server for Rust code generation models.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rust_rl.evaluation.config import UnifiedConfig
from rust_rl.evaluation.orchestration import VLLMServerManager


def main():
    parser = argparse.ArgumentParser(
        description="Start vLLM inference server for Rust code generation"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="multi_model_eval_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model ID to serve (required)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--status",
        action="store_true", 
        help="Show server status"
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop running server"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force restart if already running"
    )
    
    args = parser.parse_args()
    
    # Check config exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        config = UnifiedConfig.from_yaml(args.config)
        server_manager = VLLMServerManager(config)
        
        if args.list:
            print("Available models:")
            for model in config.models_vllm:
                print(f"  - {model.model}")
        elif args.status:
            status = server_manager.get_server_status()
            print(f"Server running: {status['running']}")
            if status['current_model']:
                print(f"Current model: {status['current_model']}")
        elif args.stop:
            server_manager.stop_server()
        elif args.model:
            success = server_manager.start_model(args.model, force_restart=args.force)
            if success:
                print(f"✅ Server started with model: {args.model}")
                print(f"📡 URL: {config.get_vllm_server_url()}")
                try:
                    server_manager.run_interactive_mode()
                except KeyboardInterrupt:
                    print("\n🛑 Stopping server...")
                    server_manager.stop_server()
            else:
                sys.exit(1)
        else:
            print("❌ Must specify --model, --list, --status, or --stop")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()