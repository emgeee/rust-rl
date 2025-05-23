#!/usr/bin/env python3
"""
Dynamic Inference Server Startup Script

Loads models on demand with interactive monitoring.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rust_rl.evaluation.config import UnifiedConfig
from rust_rl.evaluation.dynamic_model_server import DynamicModelServer, ModelLoadQueue


def main():
    parser = argparse.ArgumentParser(
        description="Start dynamic vLLM inference server in interactive mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start dynamic server in interactive mode
  python start_vllm_server.py
  
  # List available models
  python start_vllm_server.py --list
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="multi_model_eval_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    # Check config exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        config = UnifiedConfig.from_yaml(args.config)
        
        # Handle list action
        if args.list:
            print("Available models:")
            for model in config.models_vllm:
                print(f"  - {model.model}")
            return
        
        # Initialize dynamic server and start in interactive mode
        dynamic_server = DynamicModelServer(config)
        run_interactive_mode(config, dynamic_server)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def run_interactive_mode(config: UnifiedConfig, dynamic_server: DynamicModelServer):
    """Run server in interactive mode"""
    print(f"🚀 Starting Dynamic Server")
    print("=" * 50)
    
    load_queue = ModelLoadQueue(dynamic_server)
    
    print("🤖 Dynamic model server ready!")
    print(f"📡 Server will be available at: {config.get_vllm_server_url()}")
    print("💡 Models will be loaded automatically when requested")
    print("📋 Available models:")
    for model in config.models_vllm:
        print(f"   - {model.model}")
    
    print("\n🎯 To use this server:")
    print("   1. Run inference/evaluation scripts normally")
    print("   2. Models will be loaded automatically as needed")
    print("   3. Use the interactive commands below to manage the server")
    
    print("\n🖥️  Running in interactive mode")
    print("Available commands:")
    print("  status       - Show server status")
    print("  load <model> - Load a specific model")
    print("  stop         - Stop current model")
    print("  list         - List available models")
    print("  help         - Show this help")
    print("  quit         - Exit and stop server")
    print("-" * 60)
    
    try:
        while True:
            command = input("\nserver> ").strip().lower()
            
            if command in ["quit", "exit", "q"]:
                break
            elif command == "status":
                status = dynamic_server.get_status()
                print(f"Running: {status['running']}")
                if status['current_model']:
                    print(f"Current model: {status['current_model']}")
                else:
                    print("No model loaded")
                print(f"Server URL: {status['server_url']}")
            elif command.startswith("load "):
                model_id = command[5:].strip()
                if model_id:
                    print(f"Loading model: {model_id}")
                    success = load_queue.request_model(model_id, "interactive")
                    if success:
                        print(f"✅ Model loaded: {model_id}")
                    else:
                        print(f"❌ Failed to load model: {model_id}")
                else:
                    print("❌ Please specify a model ID")
            elif command == "stop":
                dynamic_server.stop_server()
                print("🛑 Server stopped")
            elif command == "list":
                models = dynamic_server.get_available_models()
                current_model = dynamic_server.get_current_model()
                print("Available models:")
                for model in models:
                    marker = " (current)" if model == current_model else ""
                    print(f"  - {model}{marker}")
            elif command in ["help", "h", "?"]:
                print("Available commands:")
                print("  status       - Show server status")
                print("  load <model> - Load a specific model")  
                print("  stop         - Stop current model")
                print("  list         - List available models")
                print("  help         - Show this help")
                print("  quit         - Exit and stop server")
            elif command == "":
                continue  # Empty input, just continue
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
    except KeyboardInterrupt:
        print("\n🛑 Exiting interactive mode...")
    except EOFError:
        print("\n🛑 Exiting interactive mode...")
    finally:
        print("🛑 Stopping server...")
        dynamic_server.stop_server()


if __name__ == "__main__":
    main()