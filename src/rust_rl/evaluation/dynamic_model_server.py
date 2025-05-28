"""
Dynamic Model Server

A server wrapper that automatically loads vLLM models on demand rather than 
requiring them to be pre-loaded. This allows clients to request any configured
model and the server will automatically load it if not already loaded.
"""

import time
import requests
import threading
import subprocess
from typing import Dict, Optional, Set
from pathlib import Path

from .config import UnifiedConfig
from .orchestration import VLLMServerManager


class DynamicModelServer:
    """
    A dynamic model server that automatically loads models on demand.
    
    This server acts as a proxy between clients and the vLLM server, automatically
    starting/stopping the vLLM server with the requested model when needed.
    """
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.server_manager = VLLMServerManager(config)
        self.current_model: Optional[str] = None
        self.model_load_lock = threading.Lock()
        
    def ensure_model_loaded(self, model_id: str) -> bool:
        """
        Ensure the specified model is loaded and ready.
        
        Args:
            model_id: The model identifier to load
            
        Returns:
            True if model is loaded and ready, False otherwise
        """
        with self.model_load_lock:
            # Check if the requested model is already loaded
            if self.current_model == model_id and self.server_manager.is_running():
                return True
            
            # Validate that the model is configured
            try:
                model_config = self.server_manager.get_model_config(model_id)
            except ValueError as e:
                print(f"❌ Model not found in configuration: {model_id}")
                return False
            
            print(f"🔄 Loading model: {model_id}")
            
            # Stop current server if running with different model
            if self.server_manager.is_running() and self.current_model != model_id:
                print(f"🛑 Stopping current model: {self.current_model}")
                self.server_manager.stop_server()
                time.sleep(2)  # Give server time to fully stop
            
            # Start server with requested model
            success = self.server_manager.start_model(model_id, force_restart=False)
            if success:
                self.current_model = model_id
                print(f"✅ Model loaded successfully: {model_id}")
                return True
            else:
                print(f"❌ Failed to load model: {model_id}")
                self.current_model = None
                return False
    
    def get_current_model(self) -> Optional[str]:
        """Get the currently loaded model"""
        return self.current_model
    
    def is_running(self) -> bool:
        """Check if the server is running and ready"""
        return self.server_manager.is_running()
    
    def is_process_alive(self) -> bool:
        """Check if server process is alive (may still be loading)"""
        return self.server_manager.is_process_alive()
    
    def get_status_summary(self) -> str:
        """Get human-readable status summary"""
        return self.server_manager.get_status_summary()
    
    def stop_server(self):
        """Stop the current server"""
        self.server_manager.stop_server()
        self.current_model = None
    
    def get_server_url(self) -> str:
        """Get the server URL"""
        return self.config.get_vllm_server_url()
    
    def get_available_models(self) -> list:
        """Get list of available models from configuration"""
        return [model.model for model in self.config.models_vllm]
    
    def get_status(self) -> dict:
        """Get comprehensive server status"""
        server_status = self.server_manager.get_server_status()
        return {
            "running": server_status["health_check"],
            "process_alive": server_status["process_alive"],
            "status": server_status["status"],
            "current_model": self.current_model,
            "models_loaded": server_status["models_loaded"],
            "server_url": self.get_server_url(),
            "available_models": self.get_available_models(),
            "server_details": server_status
        }


class ModelLoadRequest:
    """Represents a model loading request with metadata"""
    
    def __init__(self, model_id: str, requester: str = "unknown"):
        self.model_id = model_id
        self.requester = requester
        self.timestamp = time.time()
        self.status = "pending"  # pending, loading, loaded, failed
    
    def __repr__(self):
        return f"ModelLoadRequest({self.model_id}, {self.status}, {self.requester})"


class ModelLoadQueue:
    """
    Queue for managing model loading requests.
    
    Since only one model can be loaded at a time, this queue ensures
    requests are processed sequentially.
    """
    
    def __init__(self, dynamic_server: DynamicModelServer):
        self.dynamic_server = dynamic_server
        self.queue: list[ModelLoadRequest] = []
        self.processing = False
        self.queue_lock = threading.Lock()
    
    def request_model(self, model_id: str, requester: str = "unknown") -> bool:
        """
        Request a model to be loaded.
        
        Args:
            model_id: Model to load
            requester: Identifier of the requester
            
        Returns:
            True if model is already loaded or successfully loaded, False otherwise
        """
        # Check if model is already loaded
        if self.dynamic_server.get_current_model() == model_id and self.dynamic_server.is_running():
            return True
        
        # Add to queue and process
        with self.queue_lock:
            # Check if already in queue
            for req in self.queue:
                if req.model_id == model_id and req.status in ["pending", "loading"]:
                    # Wait for this request to complete
                    return self._wait_for_request(req)
            
            # Add new request
            request = ModelLoadRequest(model_id, requester)
            self.queue.append(request)
            
            # Process immediately if not already processing
            if not self.processing:
                return self._process_queue()
            else:
                # Wait for our request to be processed
                return self._wait_for_request(request)
    
    def _process_queue(self) -> bool:
        """Process the model loading queue"""
        with self.queue_lock:
            if self.processing or not self.queue:
                return True
            
            self.processing = True
            request = self.queue[0]
        
        try:
            request.status = "loading"
            success = self.dynamic_server.ensure_model_loaded(request.model_id)
            request.status = "loaded" if success else "failed"
            
            # Remove completed request
            with self.queue_lock:
                if self.queue and self.queue[0] == request:
                    self.queue.pop(0)
                self.processing = False
            
            return success
            
        except Exception as e:
            request.status = "failed"
            print(f"❌ Error processing model request: {e}")
            with self.queue_lock:
                if self.queue and self.queue[0] == request:
                    self.queue.pop(0)
                self.processing = False
            return False
    
    def _wait_for_request(self, request: ModelLoadRequest, timeout: int = 300) -> bool:
        """Wait for a specific request to complete"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request.status in ["loaded", "failed"]:
                return request.status == "loaded"
            time.sleep(1)
        
        print(f"⏰ Timeout waiting for model {request.model_id} to load")
        return False