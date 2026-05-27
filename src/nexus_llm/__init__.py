"""
Nexus LLM - Enterprise-grade 7B LLM Training Framework
"""

__version__ = "1.0.0"
__author__ = "Nexus AI Team"

from .model import NexusModel, NexusConfig
from .data import NexusTokenizer, DataProcessor
from .training import Trainer, DistributedTrainer
from .serving import InferenceEngine, ModelServer
from .utils import setup_logging, get_device_info

__all__ = [
    "NexusModel",
    "NexusConfig",
    "NexusTokenizer",
    "DataProcessor",
    "Trainer",
    "DistributedTrainer",
    "InferenceEngine",
    "ModelServer",
    "setup_logging",
    "get_device_info",
]