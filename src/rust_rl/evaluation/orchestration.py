"""
Evaluation Pipeline Orchestration

This module contains the orchestration logic for running inference, evaluation,
and visualization across multiple models. Moved from root-level scripts to
maintain clean project structure.
"""

import subprocess
import time
import signal
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import UnifiedConfig, ModelConfig
from .inference_runner import InferenceRunner
from .eval_runner import EvaluationRunner
from .multi_model_visualize import MultiModelVisualizer


class VLLMServerManager:
    """Manages vLLM server lifecycle - moved from vllm_model_server.py"""
    
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


class EvaluationOrchestrator:
    """Main orchestrator for evaluation pipeline - moved from multi_model_eval.py"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        
        self.inference_runner = InferenceRunner(config)
        self.eval_runner = EvaluationRunner(config)
        self.visualizer = MultiModelVisualizer(config)
        self.server_manager = VLLMServerManager(config)
    
    def validate_config(self, selected_models: List[str] = None):
        """Validate configuration and model selection"""
        print(f"📋 Configuration loaded from: {self.config}")
        print(f"   - Dataset: {self.config.dataset_path}")
        print(f"   - Output directory: {self.config.output_base_dir}")
        print(f"   - Evaluation tools: {', '.join(self.config.evaluation_tools)}")
        
        # Show configured models
        all_models = self.config.get_all_models()
        print(f"   - Total models configured: {len(all_models)}")
        for model_type, models in [
            ("API", self.config.models_api),
            ("vLLM", self.config.models_vllm)
        ]:
            if models:
                print(f"     - {model_type}: {', '.join([m.name for m in models])}")
        
        # Validate selected models
        if selected_models:
            available_model_names = [m.name for m in all_models]
            invalid_models = [m for m in selected_models if m not in available_model_names]
            if invalid_models:
                raise ValueError(f"Invalid model names: {', '.join(invalid_models)}")
            print(f"🎯 Selected models: {', '.join(selected_models)}")
    
    def check_vllm_server_requirements(self, selected_models: List[str] = None):
        """Check if vLLM server is needed and available"""
        all_models = self.config.get_all_models()
        if selected_models:
            all_models = [m for m in all_models if m.name in selected_models]
        
        vllm_models = [m for m in all_models if m.provider not in ["anthropic", "openai", "xai"]]
        if not vllm_models:
            return  # No vLLM models needed
        
        server_running = self.server_manager.is_running()
        print(f"🖥️  vLLM Server status: {'✅ Running' if server_running else '❌ Not running'}")
        
        if not server_running:
            model_names = [m.name for m in vllm_models]
            raise RuntimeError(
                f"vLLM server is not running but required for models: {', '.join(model_names)}\n"
                f"Start server with: python start_vllm_server.py --model <model_id>"
            )
    
    def run_inference_stage(self, selected_models: List[str] = None, force_rerun: bool = False):
        """Run inference stage"""
        print("\n" + "=" * 60)
        print("🤖 INFERENCE STAGE")
        print("=" * 60)
        
        inference_results = self.inference_runner.run_inference_for_all_models(
            force_rerun=force_rerun,
            selected_models=selected_models
        )
        
        # Report results
        successful_models = [m for m, success in inference_results.items() if success]
        failed_models = [m for m, success in inference_results.items() if not success]
        
        print(f"\n📊 Inference Results:")
        print(f"   ✅ Successful: {len(successful_models)}")
        if successful_models:
            print(f"      {', '.join(successful_models)}")
        
        if failed_models:
            print(f"   ❌ Failed: {len(failed_models)}")
            print(f"      {', '.join(failed_models)}")
        
        return inference_results
    
    def run_evaluation_stage(self, selected_models: List[str] = None, force_rerun: bool = False):
        """Run evaluation stage"""
        print("\n" + "=" * 60)
        print("🧪 EVALUATION STAGE")
        print("=" * 60)
        
        eval_results = self.eval_runner.run_evaluation_for_all_models(
            force_rerun=force_rerun,
            selected_models=selected_models
        )
        
        # Report results
        successful_evals = [m for m, success in eval_results.items() if success]
        failed_evals = [m for m, success in eval_results.items() if not success]
        
        print(f"\n📊 Evaluation Results:")
        print(f"   ✅ Successful: {len(successful_evals)}")
        if successful_evals:
            print(f"      {', '.join(successful_evals)}")
        
        if failed_evals:
            print(f"   ❌ Failed: {len(failed_evals)}")
            print(f"      {', '.join(failed_evals)}")
        
        return eval_results
    
    def run_visualization_stage(self, selected_models: List[str] = None):
        """Run visualization stage"""
        print("\n" + "=" * 60)
        print("📈 VISUALIZATION STAGE")
        print("=" * 60)
        
        # Get all completed results
        all_results = self.eval_runner.get_all_results()
        
        # Filter by selected models if specified
        if selected_models:
            all_results = {k: v for k, v in all_results.items() if k in selected_models}
        
        if not all_results:
            print("❌ No evaluation results found for visualization")
            print("   Make sure to run inference and evaluation first")
            return {}
        
        print(f"📊 Found results for {len(all_results)} models: {', '.join(all_results.keys())}")
        
        chart_paths = self.visualizer.create_all_visualizations(all_results)
        
        print("\n📈 Visualizations created:")
        for chart_name, chart_path in chart_paths.items():
            if chart_path:
                print(f"   📄 {chart_name}: {chart_path}")
        
        return chart_paths
    
    def run_full_pipeline(self, selected_models: List[str] = None, force_rerun: bool = False):
        """Run complete evaluation pipeline"""
        print("🦀 Multi-Model Rust Code Evaluation Pipeline")
        print("=" * 80)
        
        # Validate configuration
        self.validate_config(selected_models)
        
        # Check vLLM server if needed (will raise error if not available)
        self.check_vllm_server_requirements(selected_models)
        
        # Run inference
        inference_results = self.run_inference_stage(selected_models, force_rerun)
        
        # Run evaluation
        eval_results = self.run_evaluation_stage(selected_models, force_rerun)
        
        # Run visualization
        viz_results = self.run_visualization_stage(selected_models)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETE")
        print("=" * 60)
        print(f"📁 Results available in: {self.config.output_base_dir}")
        
        return {
            "inference": inference_results,
            "evaluation": eval_results, 
            "visualization": viz_results
        }