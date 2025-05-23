#!/usr/bin/env python3
"""
vLLM Model Server

This script manages a vLLM server for serving HuggingFace models with OpenAI-compatible API.
It can start/stop the server and switch between different models.
"""

import argparse
import subprocess
import sys
import time
import signal
import requests
from pathlib import Path
from typing import Optional

from src.rust_rl.evaluation.config import UnifiedConfig


class VLLMServerManager:
    """Manages vLLM server lifecycle"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.server_process: Optional[subprocess.Popen] = None
        self.current_model: Optional[str] = None
        
    def get_model_config(self, model_id: str):
        """Get model configuration by ID"""
        for model in self.config.models_vllm:
            if model.model == model_id:
                return model
        raise ValueError(f"Model {model_id} not found in vLLM models configuration")
    
    def build_vllm_command(self, model_config) -> list:
        """Build vLLM server command"""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_config.model,
            "--host", self.config.server.host,
            "--port", str(self.config.server.port),
            "--gpu-memory-utilization", str(self.config.server.gpu_memory_utilization),
        ]
        
        # Add model-specific parameters
        if model_config.max_model_len:
            cmd.extend(["--max-model-len", str(model_config.max_model_len)])
        
        # Use model-specific tensor parallel size, fallback to server default
        tensor_parallel = model_config.tensor_parallel_size or self.config.server.tensor_parallel_size
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel)])
        
        if self.config.server.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        
        return cmd
    
    def start_model(self, model_id: str, force_restart: bool = False):
        """Start vLLM server with specific model"""
        if self.is_running() and self.current_model == model_id and not force_restart:
            print(f"✅ Server already running with model: {model_id}")
            return True
            
        if self.is_running():
            print("🔄 Stopping existing server...")
            self.stop_server()
            time.sleep(2)
        
        model_config = self.get_model_config(model_id)
        cmd = self.build_vllm_command(model_config)
        
        print(f"🚀 Starting vLLM server with model: {model_id}")
        print(f"📡 Server will be available at: {self.config.get_vllm_server_url()}")
        print(f"🔧 Command: {' '.join(cmd)}")
        
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            self.current_model = model_id
            
            # Wait for server to start
            print("⏳ Waiting for server to start...")
            if self.wait_for_server_ready(timeout=120):
                print("✅ Server started successfully!")
                return True
            else:
                print("❌ Server failed to start within timeout")
                self.stop_server()
                return False
                
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop the vLLM server"""
        if self.server_process:
            print("🛑 Stopping vLLM server...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("⚠️  Server didn't stop gracefully, forcing termination...")
                self.server_process.kill()
                self.server_process.wait()
            
            self.server_process = None
            self.current_model = None
            print("✅ Server stopped")
    
    def is_running(self) -> bool:
        """Check if server is running"""
        if not self.server_process:
            return False
        
        # Check if process is still alive
        if self.server_process.poll() is not None:
            self.server_process = None
            self.current_model = None
            return False
        
        # Check if server is responding
        try:
            response = requests.get(f"{self.config.get_vllm_server_url()}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def wait_for_server_ready(self, timeout: int = 60) -> bool:
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_running():
                return True
            time.sleep(2)
        return False
    
    def get_server_status(self) -> dict:
        """Get server status information"""
        status = {
            "running": self.is_running(),
            "current_model": self.current_model,
            "url": self.config.get_vllm_server_url(),
            "process_id": self.server_process.pid if self.server_process else None
        }
        
        if status["running"]:
            try:
                # Try to get model info from server
                response = requests.get(f"{self.config.get_vllm_server_url()}/v1/models", timeout=5)
                if response.status_code == 200:
                    models = response.json()
                    status["models"] = models.get("data", [])
            except:
                pass
        
        return status
    
    def run_interactive_mode(self):
        """Run server in interactive mode with output"""
        if not self.server_process:
            print("❌ No server process running")
            return
        
        print(f"🖥️  Running in interactive mode (Ctrl+C to stop)")
        print(f"📡 Server URL: {self.config.get_vllm_server_url()}")
        print(f"🤖 Current model: {self.current_model}")
        print("-" * 60)
        
        try:
            # Stream server output
            for line in self.server_process.stdout:
                print(line.rstrip())
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal, stopping server...")
            self.stop_server()


def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\n🛑 Received interrupt signal, shutting down...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Model Server Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with specific model
  python vllm_model_server.py --config multi_model_eval_config.yaml --model "Qwen/Qwen2.5-Coder-7B-Instruct"

  # List available models
  python vllm_model_server.py --config multi_model_eval_config.yaml --list

  # Check server status
  python vllm_model_server.py --config multi_model_eval_config.yaml --status

  # Stop running server
  python vllm_model_server.py --config multi_model_eval_config.yaml --stop
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="multi_model_eval_config.yaml",
        help="Path to unified configuration YAML file"
    )
    
    # Actions
    parser.add_argument(
        "--model",
        type=str,
        help="Model ID to serve (starts server)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available vLLM models"
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
    
    # Options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force restart if server already running"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode (background)"
    )
    
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"❌ Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Load configuration
        print(f"📋 Loading configuration from: {args.config}")
        config = UnifiedConfig.from_yaml(args.config)
        server_manager = VLLMServerManager(config)
        
        # Execute requested action
        if args.list:
            print("📋 Available vLLM models:")
            for i, model in enumerate(config.models_vllm, 1):
                print(f"  {i}. {model.model}")
                if model.max_model_len:
                    print(f"     Max length: {model.max_model_len}")
                if model.tensor_parallel_size:
                    print(f"     Tensor parallel: {model.tensor_parallel_size}")
        
        elif args.status:
            status = server_manager.get_server_status()
            print("📊 Server Status:")
            print(f"  Running: {'✅ Yes' if status['running'] else '❌ No'}")
            print(f"  URL: {status['url']}")
            if status['current_model']:
                print(f"  Current model: {status['current_model']}")
            if status['process_id']:
                print(f"  Process ID: {status['process_id']}")
        
        elif args.stop:
            if server_manager.is_running():
                server_manager.stop_server()
            else:
                print("ℹ️  No server running")
        
        elif args.model:
            # Validate model exists
            try:
                server_manager.get_model_config(args.model)
            except ValueError as e:
                print(f"❌ Error: {e}")
                print("💡 Use --list to see available models")
                sys.exit(1)
            
            # Start server
            success = server_manager.start_model(args.model, force_restart=args.force)
            if success:
                if args.daemon:
                    print("🏃 Server running in daemon mode")
                else:
                    server_manager.run_interactive_mode()
            else:
                sys.exit(1)
        
        else:
            print("❌ Error: No action specified")
            print("💡 Use --help for usage information")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()