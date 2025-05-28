"""
Configuration management for multi-model evaluation
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Server-specific configuration"""
    host: str = "localhost"
    port: int = 8000
    gpu_memory_utilization: float = 0.9
    enable_chunked_prefill: bool = True
    tensor_parallel_size: int = 1  # Default for all models


@dataclass
class ModelConfig:
    """Configuration for a single model"""
    model: str  # Primary identifier
    provider: Optional[str] = None  # For API models: anthropic, openai, xai
    max_model_len: Optional[int] = None  # For vLLM models
    tensor_parallel_size: Optional[int] = None  # For vLLM models (overrides server default)
    
    @property
    def name(self) -> str:
        """Generate display name from model identifier"""
        return self.model.lower().replace("/", "-").replace("_", "-")
    
    @property
    def model_id(self) -> str:
        """Alias for model for backward compatibility"""
        return self.model
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class UnifiedConfig:
    """Unified configuration for both server and client"""
    server: ServerConfig
    models_api: List[ModelConfig]
    models_vllm: List[ModelConfig]
    generation_params: Dict[str, Any]
    dataset_path: str
    output_base_dir: str
    evaluation_tools: List[str]
    save_every: int
    eval_dataset_rows: Optional[int] = None
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "UnifiedConfig":
        """Load unified configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Parse server configuration
        server_data = config_data["server"]
        # Allow environment variable override for server host
        host = os.getenv("VLLM_SERVER_HOST", server_data.get("host", "localhost"))
        port = int(os.getenv("VLLM_SERVER_PORT", server_data.get("port", 8000)))
        
        server_config = ServerConfig(
            host=host,
            port=port,
            gpu_memory_utilization=server_data["runtime"]["gpu_memory_utilization"],
            enable_chunked_prefill=server_data["runtime"]["enable_chunked_prefill"],
            tensor_parallel_size=server_data["runtime"]["tensor_parallel_size"]
        )
        
        # Parse model configurations
        models_api = [
            ModelConfig(model=model["model"], provider=model["provider"])
            for model in config_data["models"]["api"]
        ]
        
        models_vllm = [
            ModelConfig(
                model=model["model"],
                max_model_len=model.get("max_model_len"),
                tensor_parallel_size=model.get("tensor_parallel_size")
            )
            for model in config_data["models"]["vllm"] or []
        ]
        
        return cls(
            server=server_config,
            models_api=models_api,
            models_vllm=models_vllm,
            generation_params=config_data["generation_params"],
            dataset_path=config_data["dataset"]["path"],
            output_base_dir=config_data["output"]["base_dir"],
            evaluation_tools=config_data["evaluation"]["tools"],
            save_every=config_data["evaluation"]["save_every"],
            eval_dataset_rows=config_data["dataset"].get("eval_rows")
        )
    
    def get_all_models(self) -> List[ModelConfig]:
        """Get all model configurations"""
        return self.models_api + self.models_vllm
    
    def get_output_dir(self, model_name: str) -> Path:
        """Get output directory for a specific model"""
        return Path(self.output_base_dir) / model_name
    
    def get_vllm_server_url(self) -> str:
        """Get vLLM server URL"""
        return f"http://{self.server.host}:{self.server.port}"


# Backward compatibility alias
EvaluationConfig = UnifiedConfig