#!/usr/bin/env python3
"""
vLLM Inference Server Startup Script

Starts vLLM server with the first configured model.
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rust_rl.evaluation.config import UnifiedConfig
from rust_rl.evaluation.dynamic_model_server import DynamicModelServer, ModelLoadQueue
from rust_rl.evaluation.status_server import StatusServer


def main():
    parser = argparse.ArgumentParser(
        description="Start vLLM inference server with the first configured model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start vLLM server with first configured model
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
        dynamic_server = DynamicModelServer(config)
        
        # Handle list action
        if args.list:
            print("Available models:")
            for model in config.models_vllm:
                print(f"  - {model.model}")
            return
        
        
        
        # Start server with immediate availability
        start_server_immediate(config, dynamic_server)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def start_server_immediate(config: UnifiedConfig, dynamic_server: DynamicModelServer):
    """Start vLLM server with the first available model"""
    print(f"🚀 Starting vLLM Server")
    print("=" * 50)
    
    # Start status server first
    status_server = StatusServer(dynamic_server, port=8001)
    status_server.start()
    
    # Check if vLLM is installed
    try:
        import vllm
        print("✅ vLLM package found")
    except ImportError:
        print("❌ vLLM package not installed!")
        print("💡 Install with: pip install vllm")
        print("   Or add it to pyproject.toml dependencies")
        sys.exit(1)
    
    # Get first available vLLM model
    if not config.models_vllm:
        print("❌ No vLLM models configured in config file")
        sys.exit(1)
    
    default_model = config.models_vllm[0].model
    server_url = config.get_vllm_server_url()
    
    print(f"🤖 Loading model: {default_model}")
    print(f"📡 Server URL: {server_url}")
    
    # Start the server with the default model
    load_queue = ModelLoadQueue(dynamic_server)
    success = load_queue.request_model(default_model, "startup")
    
    if success:
        print(f"✅ Model loaded successfully: {default_model}")
        print(f"🌐 Server ready at: {server_url}")
        print("\n🎯 Server Management:")
        print(f"   vLLM health:   curl {server_url}/health")
        print(f"   Status info:   curl http://localhost:8001/status")
        print(f"   Quick health:  curl http://localhost:8001/health")
        print("   Stop server:   Use Ctrl+C or kill the vLLM process")
        print("\n✨ vLLM server is ready to serve requests!")
        
        # Keep the process alive to maintain the server
        try:
            print("\n💡 Server is running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            status_server.stop()
            dynamic_server.stop_server()
            print("✅ Server stopped")
    else:
        print(f"❌ Failed to start server with model: {default_model}")
        status_server.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()