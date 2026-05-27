"""
Utility Functions for Nexus-7B
"""

import os
import sys
import logging
import json
import yaml
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

import torch
import numpy as np


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
):
    """Setup logging configuration."""
    
    # Create logger
    logger = logging.getLogger("nexus_llm")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_device_info() -> Dict[str, Any]:
    """Get device information for training/inference."""
    
    info = {
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [],
        "cuda_version": None,
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            
            info["cuda_devices"].append({
                "index": i,
                "name": device_props.name,
                "total_memory": device_props.total_memory,
                "compute_capability": device_props.major,
            })
    
    return info


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to file."""
    
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def estimate_memory_usage(
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    precision: str = "bf16",
) -> Dict[str, int]:
    """Estimate model memory usage."""
    
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    
    # Estimate parameters
    # Embedding: vocab_size * hidden_size
    embedding_params = vocab_size * hidden_size
    
    # Each layer: attention (4 * hidden_size^2) + MLP (3 * hidden_size * intermediate_size)
    attention_params = 4 * hidden_size * hidden_size
    mlp_params = 3 * hidden_size * (4 * hidden_size)  # intermediate_size ≈ 4 * hidden_size
    
    layer_params = attention_params + mlp_params
    total_layer_params = num_layers * layer_params
    
    # Total params (excluding biases and layer norms)
    total_params = embedding_params + total_layer_params
    
    # Memory estimation
    bytes_per = bytes_per_param.get(precision, 2)
    
    model_memory = total_params * bytes_per
    
    # KV cache estimation (for inference)
    # 2 * num_layers * hidden_size * max_seq_length * bytes_per
    kv_cache_memory = 2 * num_layers * hidden_size * 8192 * bytes_per
    
    return {
        "total_params": total_params,
        "model_memory_bytes": model_memory,
        "model_memory_gb": model_memory / (1024 ** 3),
        "kv_cache_memory_bytes": kv_cache_memory,
        "kv_cache_memory_gb": kv_cache_memory / (1024 ** 3),
        "total_memory_gb": (model_memory + kv_cache_memory) / (1024 ** 3),
    }


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_memory(bytes: int) -> str:
    """Format memory in human-readable format."""
    
    if bytes < 1024:
        return f"{bytes}B"
    elif bytes < 1024 ** 2:
        return f"{bytes / 1024:.2f}KB"
    elif bytes < 1024 ** 3:
        return f"{bytes / (1024 ** 2):.2f}MB"
    else:
        return f"{bytes / (1024 ** 3):.2f}GB"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def get_gradient_norm(model: torch.nn.Module) -> float:
    """Get total gradient norm."""
    
    total_norm = 0.0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    return total_norm ** 0.5


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_timestamp() -> str:
    """Get current timestamp string."""
    
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_dir(
    base_dir: str,
    experiment_name: str,
) -> Path:
    """Create experiment directory with timestamp."""
    
    timestamp = get_timestamp()
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start timer."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """Stop timer and return elapsed time."""
        self.end_time = time.time()
        return self.elapsed
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time."""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time or time.time()
        return end - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        elapsed = self.stop()
        print(f"{self.name}: {format_time(elapsed)}")


class AverageMeter:
    """Maintain running average of values."""
    
    def __init__(self, name: str = "Average"):
        self.name = name
        self.sum = 0.0
        self.count = 0
    
    def update(self, value: float, n: int = 1):
        """Update average with new value."""
        self.sum += value * n
        self.count += n
    
    @property
    def average(self) -> float:
        """Get current average."""
        if self.count == 0:
            return 0.0
        return self.sum / self.count
    
    def reset(self):
        """Reset average."""
        self.sum = 0.0
        self.count = 0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.average:.4f}"


class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(
        self,
        total: int,
        description: str = "Progress",
        log_interval: int = 100,
    ):
        self.total = total
        self.description = description
        self.log_interval = log_interval
        
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        
        if self.current % self.log_interval == 0 or self.current == self.total:
            elapsed = time.time() - self.start_time
            remaining = elapsed / self.current * (self.total - self.current)
            
            print(
                f"{self.description}: {self.current}/{self.total} "
                f"({self.current/self.total*100:.1f}%) "
                f"Elapsed: {format_time(elapsed)} "
                f"Remaining: {format_time(remaining)}"
            )
    
    @property
    def progress(self) -> float:
        """Get progress percentage."""
        return self.current / self.total * 100


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed."""
    
    dependencies = {
        "torch": True,
        "transformers": False,
        "accelerate": False,
        "deepspeed": False,
        "flash_attn": False,
        "sentencepiece": False,
        "wandb": False,
        "fastapi": False,
        "prometheus_client": False,
        "bitsandbytes": False,
    }
    
    try:
        import transformers
        dependencies["transformers"] = True
    except ImportError:
        pass
    
    try:
        import accelerate
        dependencies["accelerate"] = True
    except ImportError:
        pass
    
    try:
        import deepspeed
        dependencies["deepspeed"] = True
    except ImportError:
        pass
    
    try:
        import flash_attn
        dependencies["flash_attn"] = True
    except ImportError:
        pass
    
    try:
        import sentencepiece
        dependencies["sentencepiece"] = True
    except ImportError:
        pass
    
    try:
        import wandb
        dependencies["wandb"] = True
    except ImportError:
        pass
    
    try:
        import fastapi
        dependencies["fastapi"] = True
    except ImportError:
        pass
    
    try:
        import prometheus_client
        dependencies["prometheus_client"] = True
    except ImportError:
        pass
    
    try:
        import bitsandbytes
        dependencies["bitsandbytes"] = True
    except ImportError:
        pass
    
    return dependencies


def print_system_info():
    """Print system information."""
    
    print("=" * 50)
    print("Nexus-7B System Information")
    print("=" * 50)
    
    # Device info
    device_info = get_device_info()
    
    print(f"CPU cores: {device_info['cpu_count']}")
    print(f"CUDA available: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        print(f"CUDA version: {device_info['cuda_version']}")
        
        for gpu in device_info['cuda_devices']:
            print(f"  GPU {gpu['index']}: {gpu['name']}")
            print(f"    Memory: {format_memory(gpu['total_memory'])}")
            print(f"    Compute: {gpu['compute_capability']}")
    
    # Dependencies
    print("\nDependencies:")
    deps = check_dependencies()
    
    for dep, installed in deps.items():
        status = "✓" if installed else "✗"
        print(f"  {dep}: {status}")
    
    print("=" * 50)