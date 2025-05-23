#!/usr/bin/env python3
"""
Unified Inference Server Startup Script

Supports both traditional (single model) and dynamic (load on demand) modes.
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rust_rl.evaluation.config import UnifiedConfig
from rust_rl.evaluation.orchestration import VLLMServerManager
from rust_rl.evaluation.dynamic_model_server import DynamicModelServer, ModelLoadQueue


def main():
    parser = argparse.ArgumentParser(
        description="Start vLLM inference server (traditional or dynamic mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Traditional mode - start with specific model
  python start_server.py --model "Qwen/Qwen2.5-Coder-7B-Instruct"
  
  # Dynamic mode - load models on demand
  python start_server.py --dynamic
  
  # Dynamic mode with interactive monitoring
  python start_server.py --dynamic --interactive
  
  # Check server status
  python start_server.py --status
  
  # List available models
  python start_server.py --list
  
  # Stop running server
  python start_server.py --stop
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="multi_model_eval_config.yaml",
        help="Path to configuration file"
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--model",
        type=str,
        help="Start traditional server with specific model"
    )
    mode_group.add_argument(
        "--dynamic",
        action="store_true",
        help="Start dynamic server (loads models on demand)"
    )
    
    # Actions
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    action_group.add_argument(
        "--status",
        action="store_true", 
        help="Show server status"
    )
    action_group.add_argument(
        "--stop",
        action="store_true",
        help="Stop running server"
    )
    
    # Options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force restart if already running"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (only for dynamic mode)"
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
        
        # Initialize servers
        traditional_server = VLLMServerManager(config)
        dynamic_server = DynamicModelServer(config) if args.dynamic else None
        
        # Handle status action
        if args.status:
            if args.dynamic and dynamic_server:
                status = dynamic_server.get_status()
                print(f"Mode: Dynamic")
                print(f"Server running: {status['running']}")
                if status['current_model']:
                    print(f"Current model: {status['current_model']}")
                print(f"Server URL: {status['server_url']}")
                print(f"Available models: {len(status['available_models'])}")
            else:
                status = traditional_server.get_server_status()
                print(f"Mode: Traditional")
                print(f"Server running: {status['running']}")
                if status['current_model']:
                    print(f"Current model: {status['current_model']}")
                print(f"Server URL: {status['url']}")
            return
        
        # Handle stop action
        if args.stop:
            if dynamic_server:
                dynamic_server.stop_server()
            else:
                traditional_server.stop_server()
            print("🛑 Server stopped")
            return
        
        # Start server modes
        if args.dynamic:
            run_dynamic_mode(config, dynamic_server, args.interactive)
        elif args.model:
            run_traditional_mode(config, traditional_server, args.model, args.force)
        else:
            print("❌ Must specify --model <model_id> for traditional mode or --dynamic for dynamic mode")
            print("Use --help for usage examples")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def run_traditional_mode(config: UnifiedConfig, server_manager: VLLMServerManager, model_id: str, force_restart: bool):
    """Run server in traditional mode with specific model"""
    print(f"🚀 Starting Traditional Server Mode")
    print(f"Model: {model_id}")
    print("=" * 50)
    
    success = server_manager.start_model(model_id, force_restart=force_restart)
    if success:
        print(f"✅ Server started with model: {model_id}")
        print(f"📡 URL: {config.get_vllm_server_url()}")
        print("\n💡 To use this server:")
        print("   - Run evaluation scripts normally")
        print("   - Only the loaded model will be available")
        print("   - Use Ctrl+C to stop")
        
        try:
            server_manager.run_interactive_mode()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            server_manager.stop_server()
    else:
        sys.exit(1)


def run_dynamic_mode(config: UnifiedConfig, dynamic_server: DynamicModelServer, interactive: bool):
    """Run server in dynamic mode"""
    print(f"🚀 Starting Dynamic Server Mode")
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
    print("   3. Use --interactive for live monitoring")
    
    if interactive:
        run_interactive_mode(dynamic_server, load_queue)
    else:
        print("\n🖥️  Dynamic server running in background mode")
        print("💡 Use --interactive for real-time monitoring")
        print("💡 Use --status to check server status")
        print("💡 Use --stop to stop the server")
        print("\nPress Ctrl+C to stop...")
        
        try:
            # Keep script running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping dynamic server...")
            dynamic_server.stop_server()


def run_interactive_mode(dynamic_server: DynamicModelServer, load_queue: ModelLoadQueue):
    """Run dynamic server in interactive monitoring mode"""
    print("\n🖥️  Running in interactive mode")
    print("Available commands:")
    print("  status       - Show server status")
    print("  load <model> - Load a specific model")
    print("  stop         - Stop current model")
    print("  list         - List available models")
    print("  help         - Show this help")
    print("  quit         - Exit interactive mode")
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
                print("  quit         - Exit interactive mode")
            elif command == "":
                continue  # Empty input, just continue
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
    except KeyboardInterrupt:
        print("\n🛑 Exiting interactive mode...")
    except EOFError:
        print("\n🛑 Exiting interactive mode...")
    finally:
        dynamic_server.stop_server()


if __name__ == "__main__":
    main()