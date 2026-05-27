"""
Nexus LLM - Enterprise-grade 7B LLM Training Framework
Enterprise-ready LLM framework with RLHF, Agent, Compression and Incremental Learning
"""

__version__ = "2.1.0"
__author__ = "Nexus AI Team"

# Core modules
from .model import NexusModel, NexusConfig, NexusForCausalLM
from .data import NexusTokenizer, DataProcessor, DataConfig
from .training import Trainer, DistributedTrainer, TrainingConfig
from .serving import InferenceEngine, ModelServer, InferenceConfig
from .utils import setup_logging, get_device_info

# Advanced modules
from .rlhf import RLHFConfig, RewardModel, PPOTrainer, RewardModelTrainer, create_rlhf_pipeline
from .agent import (
    ReActAgent, FunctionCallingAgent, MultiAgentCoordinator,
    ToolRegistry, Tool, ToolCall, Message, MessageRole,
    BaseTool, Calculator, WebSearch, KnowledgeBase
)
from .compression import (
    DistillationConfig, PruningConfig, QuantizationConfig,
    KnowledgeDistiller, StructuredPruner, ModelCompressor
)
from .incremental import (
    IncrementalConfig, ExperienceReplayBuffer, EWCRegularizer,
    AdapterModule, AdapterModel, IncrementalTrainer
)

# OpenAI Compatible API
from .openai_api import (
    OpenAICompatibleServer,
    OpenAIServerConfig,
    create_openai_server,
    run_openai_server,
)

__all__ = [
    # Core
    "NexusModel",
    "NexusConfig",
    "NexusForCausalLM",
    "NexusTokenizer",
    "DataProcessor",
    "DataConfig",
    "Trainer",
    "DistributedTrainer",
    "TrainingConfig",
    "InferenceEngine",
    "ModelServer",
    "InferenceConfig",
    "setup_logging",
    "get_device_info",

    # RLHF
    "RLHFConfig",
    "RewardModel",
    "PPOTrainer",
    "RewardModelTrainer",
    "create_rlhf_pipeline",

    # Agent
    "ReActAgent",
    "FunctionCallingAgent",
    "MultiAgentCoordinator",
    "ToolRegistry",
    "Tool",
    "ToolCall",
    "Message",
    "MessageRole",
    "BaseTool",
    "Calculator",
    "WebSearch",
    "KnowledgeBase",

    # Compression
    "DistillationConfig",
    "PruningConfig",
    "QuantizationConfig",
    "KnowledgeDistiller",
    "StructuredPruner",
    "ModelCompressor",

    # Incremental Learning
    "IncrementalConfig",
    "ExperienceReplayBuffer",
    "EWCRegularizer",
    "AdapterModule",
    "AdapterModel",
    "IncrementalTrainer",

    # OpenAI Compatible API
    "OpenAICompatibleServer",
    "OpenAIServerConfig",
    "create_openai_server",
    "run_openai_server",
]
