"""
Training System with Distributed Training Support
"""

import os
import sys
import json
import logging
import time
import math
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..model import NexusConfig, NexusForCausalLM
from ..data import NexusTokenizer, DataProcessor, DataConfig


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Output directories
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    
    # Batch settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    micro_batch_size: int = 1
    
    # Training steps
    max_steps: int = 100000
    save_steps: int = 1000
    eval_steps: int = 500
    log_steps: int = 10
    warmup_steps: int = 1000
    
    # Learning rate scheduler
    lr_scheduler: str = "cosine"
    min_lr_ratio: float = 0.1
    
    # Precision
    precision: str = "bf16"  # fp16, bf16, fp32
    
    # Gradient checkpointing
    gradient_checkpointing: bool = True
    
    # Distributed training
    distributed_backend: str = "nccl"
    distributed_strategy: str = "deepspeed"  # or "fsdp", "ddp"
    
    # DeepSpeed config
    deepspeed_config: Optional[str] = None
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "nexus-llm"
    wandb_entity: Optional[str] = None
    use_tensorboard: bool = True
    
    # Checkpointing
    save_total_limit: int = 5
    save_best_only: bool = True
    metric_for_best: str = "eval_loss"
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Seed
    seed: int = 42
    
    # Evaluation
    eval_batch_size: int = 8
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.001
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class LRScheduler:
    """Learning rate scheduler with warmup and decay."""
    
    def __init__(self, optimizer: optim.Optimizer, config: TrainingConfig, total_steps: int):
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        
        self.base_lr = config.learning_rate
        self.min_lr = config.learning_rate * config.min_lr_ratio
        self.warmup_steps = config.warmup_steps
        
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Decay phase
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            
            if self.config.lr_scheduler == "cosine":
                lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            elif self.config.lr_scheduler == "linear":
                lr = self.base_lr - (self.base_lr - self.min_lr) * progress
            elif self.config.lr_scheduler == "constant":
                lr = self.base_lr
            else:
                lr = self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    """Base trainer for Nexus-7B."""
    
    def __init__(
        self,
        model: NexusForCausalLM,
        tokenizer: NexusTokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: TrainingConfig = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup scheduler
        total_steps = self.config.max_steps
        self.scheduler = LRScheduler(self.optimizer, self.config, total_steps)
        
        # Setup precision
        self._setup_precision()
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.model.enable_gradient_checkpointing()
        
        # Training state
        self.global_step = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # Resume from checkpoint
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)
    
    def _setup_directories(self):
        """Setup output directories."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging systems."""
        # Setup wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.to_dict(),
                name=f"nexus-7b-{time.strftime('%Y%m%d-%H%M%S')}",
            )
        
        # Setup tensorboard
        if self.config.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(self.config.log_dir)
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        # Filter parameters that require gradients
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in ["bias", "LayerNorm", "layer_norm", "rms_norm"])],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in ["bias", "LayerNorm", "layer_norm", "rms_norm"])],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=1e-8,
        )
    
    def _setup_precision(self):
        """Setup mixed precision training."""
        self.scaler = None
        
        if self.config.precision == "fp16":
            self.scaler = torch.cuda.amp.GradScaler()
            self.dtype = torch.float16
        elif self.config.precision == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total steps: {self.config.max_steps}")
        
        self.model.train()
        
        progress_bar = tqdm(
            total=self.config.max_steps,
            desc="Training",
            initial=self.global_step,
        )
        
        while self.global_step < self.config.max_steps:
            for batch in self.train_dataloader:
                # Check if we should stop
                if self.global_step >= self.config.max_steps:
                    break
                
                # Training step
                loss = self._training_step(batch)
                
                # Logging
                if self.global_step % self.config.log_steps == 0:
                    self._log_metrics(loss)
                
                # Evaluation
                if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                    eval_loss = self._evaluate()
                    self._log_eval_metrics(eval_loss)
                    
                    # Check for best model
                    if eval_loss < self.best_metric - self.config.early_stopping_threshold:
                        self.best_metric = eval_loss
                        self.patience_counter = 0
                        self._save_checkpoint("best")
                    else:
                        self.patience_counter += 1
                    
                    # Early stopping
                    if self.patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at step {self.global_step}")
                        break
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}")
                
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{self.scheduler.get_lr():.6f}"})
        
        progress_bar.close()
        
        # Final save
        self._save_checkpoint("final")
        
        logger.info("Training completed!")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute single training step."""
        # Move batch to device
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(dtype=self.dtype):
            outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            
            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Scheduler step
            self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def _evaluate(self) -> float:
        """Evaluate model on validation set."""
        logger.info("Running evaluation...")
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    outputs = self.model(**batch)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        self.model.train()
        
        return avg_loss
    
    def _log_metrics(self, loss: float):
        """Log training metrics."""
        metrics = {
            "train/loss": loss,
            "train/learning_rate": self.scheduler.get_lr(),
            "train/global_step": self.global_step,
        }
        
        # Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics)
        
        # Tensorboard
        if self.config.use_tensorboard:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, self.global_step)
        
        logger.info(f"Step {self.global_step}: loss={loss:.4f}, lr={self.scheduler.get_lr():.6f}")
    
    def _log_eval_metrics(self, eval_loss: float):
        """Log evaluation metrics."""
        metrics = {
            "eval/loss": eval_loss,
            "eval/perplexity": math.exp(eval_loss),
            "eval/global_step": self.global_step,
        }
        
        # Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics)
        
        # Tensorboard
        if self.config.use_tensorboard:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, self.global_step)
        
        logger.info(f"Evaluation at step {self.global_step}: loss={eval_loss:.4f}, perplexity={math.exp(eval_loss):.2f}")
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(checkpoint_path))
        
        # Save optimizer and scheduler
        torch.save({
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state": {
                "current_step": self.scheduler.current_step,
            },
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "config": self.config.to_dict(),
        }, checkpoint_path / "training_state.pt")
        
        # Save tokenizer
        tokenizer_path = checkpoint_path / "tokenizer"
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        # Copy tokenizer files
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Manage checkpoint limit
        self._manage_checkpoints()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load model
        self.model = NexusForCausalLM.from_pretrained(str(checkpoint_path))
        
        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path)
            self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
            self.scheduler.current_step = training_state["scheduler_state"]["current_step"]
            self.global_step = training_state["global_step"]
            self.best_metric = training_state["best_metric"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def _manage_checkpoints(self):
        """Manage checkpoint limit."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda x: int(x.name.split("_")[1])
        )
        
        # Remove old checkpoints
        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            for file in oldest.iterdir():
                file.unlink()
            oldest.rmdir()
            logger.info(f"Removed old checkpoint: {oldest}")


class DistributedTrainer(Trainer):
    """Distributed trainer with DeepSpeed/FSDP support."""
    
    def __init__(
        self,
        model: NexusForCausalLM,
        tokenizer: NexusTokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: TrainingConfig = None,
    ):
        # Initialize distributed training
        self._setup_distributed()
        
        super().__init__(model, tokenizer, train_dataloader, eval_dataloader, config)
        
        # Setup distributed model
        self._setup_distributed_model()
    
    def _setup_distributed(self):
        """Setup distributed training environment."""
        # Get distributed info
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        
        # Initialize process group
        if self.world_size > 1:
            dist.init_process_group(
                backend=self.config.distributed_backend,
                init_method="env://",
            )
            
            # Set device
            torch.cuda.set_device(self.local_rank)
        
        self.is_main_process = self.rank == 0
        
        logger.info(f"Distributed setup: rank={self.rank}, local_rank={self.local_rank}, world_size={self.world_size}")
    
    def _setup_distributed_model(self):
        """Setup model for distributed training."""
        if self.config.distributed_strategy == "deepspeed" and DEEPSPEED_AVAILABLE:
            self._setup_deepspeed()
        elif self.config.distributed_strategy == "fsdp":
            self._setup_fsdp()
        elif self.config.distributed_strategy == "ddp":
            self._setup_ddp()
    
    def _setup_deepspeed(self):
        """Setup DeepSpeed training."""
        # Load DeepSpeed config
        if self.config.deepspeed_config:
            with open(self.config.deepspeed_config) as f:
                ds_config = json.load(f)
        else:
            ds_config = self._create_deepspeed_config()
        
        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config=ds_config,
        )
        
        self.model = self.model_engine
        
        logger.info("DeepSpeed initialized successfully")
    
    def _create_deepspeed_config(self) -> Dict:
        """Create default DeepSpeed configuration."""
        return {
            "train_batch_size": self.config.batch_size * self.config.gradient_accumulation_steps * self.world_size,
            "train_micro_batch_size_per_gpu": self.config.micro_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.config.learning_rate,
                    "betas": [self.config.beta1, self.config.beta2],
                    "eps": 1e-8,
                    "weight_decay": self.config.weight_decay,
                }
            },
            
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "total_num_steps": self.config.max_steps,
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.config.learning_rate,
                    "warmup_num_steps": self.config.warmup_steps,
                }
            },
            
            "gradient_clipping": self.config.max_grad_norm,
            
            "bf16": {
                "enabled": self.config.precision == "bf16",
            },
            
            "fp16": {
                "enabled": self.config.precision == "fp16",
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True,
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
            },
            
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "contiguous_memory_optimization": False,
                "synchronize_checkpoint_boundary": False,
            },
        }
    
    def _setup_fsdp(self):
        """Setup Fully Sharded Data Parallel training."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        
        # Mixed precision policy
        mixed_precision = MixedPrecision(
            param_dtype=self.dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        
        # Auto wrap policy for transformer layers
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_names=["NexusDecoderLayer"],
        )
        
        # Initialize FSDP
        self.model = FSDP(
            self.model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
        )
        
        logger.info("FSDP initialized successfully")
    
    def _setup_ddp(self):
        """Setup DistributedDataParallel training."""
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
        )
        
        logger.info("DDP initialized successfully")
    
    def _log_metrics(self, loss: float):
        """Log training metrics (only on main process)."""
        if not self.is_main_process:
            return
        
        super()._log_metrics(loss)
    
    def _save_checkpoint(self, name: str):
        """Save checkpoint (only on main process)."""
        if not self.is_main_process:
            return
        
        checkpoint_path = Path(self.config.checkpoint_dir) / name
        
        if self.config.distributed_strategy == "deepspeed":
            # DeepSpeed has its own checkpoint saving
            self.model.save_checkpoint(str(checkpoint_path))
        else:
            super()._save_checkpoint(name)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for distributed training."""
        if self.config.distributed_strategy == "deepspeed":
            self.model.load_checkpoint(checkpoint_path)
        else:
            super()._load_checkpoint(checkpoint_path)


class FineTuningTrainer(Trainer):
    """Trainer for fine-tuning with LoRA/QLoRA support."""
    
    def __init__(
        self,
        model: NexusForCausalLM,
        tokenizer: NexusTokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: TrainingConfig = None,
        lora_config: Optional[Dict] = None,
    ):
        self.lora_config = lora_config
        
        # Apply LoRA if configured
        if lora_config:
            self._apply_lora(model, lora_config)
        
        super().__init__(model, tokenizer, train_dataloader, eval_dataloader, config)
    
    def _apply_lora(self, model: NexusForCausalLM, lora_config: Dict):
        """Apply LoRA to model."""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # Prepare model for LoRA
            if lora_config.get("quantization"):
                model = prepare_model_for_kbit_training(model)
            
            # Create LoRA config
            peft_config = LoraConfig(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("alpha", 32),
                target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                lora_dropout=lora_config.get("dropout", 0.05),
                bias=lora_config.get("bias", "none"),
                task_type="CAUSAL_LM",
            )
            
            # Apply LoRA
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            
            self.model = model
            self.is_lora = True
            
        except ImportError:
            logger.warning("PEFT library not installed, skipping LoRA")
            self.is_lora = False
    
    def _save_checkpoint(self, name: str):
        """Save checkpoint with LoRA weights."""
        checkpoint_path = Path(self.config.checkpoint_dir) / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        if self.is_lora:
            # Save LoRA weights only
            self.model.save_pretrained(str(checkpoint_path))
        else:
            super()._save_checkpoint(name)


def create_trainer(
    model_config: NexusConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
    distributed: bool = False,
    lora_config: Optional[Dict] = None,
) -> Union[Trainer, DistributedTrainer, FineTuningTrainer]:
    """Factory function to create appropriate trainer."""
    
    # Create model
    model = NexusForCausalLM(model_config)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create tokenizer
    tokenizer = NexusTokenizer(data_config)
    
    # Create data processor
    processor = DataProcessor(data_config, tokenizer)
    datasets = processor.create_datasets()
    dataloaders = processor.create_dataloaders(datasets, training_config.batch_size)
    
    # Select trainer type
    if distributed:
        trainer = DistributedTrainer(
            model, tokenizer,
            dataloaders['train'], dataloaders.get('valid'),
            training_config,
        )
    elif lora_config:
        trainer = FineTuningTrainer(
            model, tokenizer,
            dataloaders['train'], dataloaders.get('valid'),
            training_config,
            lora_config,
        )
    else:
        trainer = Trainer(
            model, tokenizer,
            dataloaders['train'], dataloaders.get('valid'),
            training_config,
        )
    
    return trainer