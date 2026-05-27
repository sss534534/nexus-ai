"""
Model Checkpoint Management with Safetensors Support
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, asdict
import shutil
import hashlib

import torch
import torch.nn as nn
from torch.optim import Optimizer

# Try to import safetensors
try:
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logging.warning("safetensors not available, falling back to PyTorch format")

from . import NexusConfig, NexusForCausalLM


logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for model checkpoint."""
    
    # Model info
    model_name: str
    model_version: str = "1.0.0"
    model_config: Dict[str, Any] = None
    
    # Training info
    training_step: int = 0
    training_epoch: int = 0
    global_step: int = 0
    
    # Metrics
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    perplexity: Optional[float] = None
    
    # Timestamps
    created_at: str = ""
    last_modified: str = ""
    
    # Checksum
    checksum: Optional[str] = None
    
    # Tags
    tags: List[str] = None
    
    def __post_init__(self):
        from datetime import datetime
        
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.last_modified:
            self.last_modified = datetime.utcnow().isoformat()
        if self.tags is None:
            self.tags = []
        if self.model_config is None:
            self.model_config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """Manager for model checkpoints with safetensors support."""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        save_safetensors: bool = True,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        encryption_key: Optional[bytes] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_safetensors = save_safetensors and SAFETENSORS_AVAILABLE
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.encryption_key = encryption_key
        
        # Track checkpoints
        self.checkpoints: List[Path] = []
        self._scan_existing_checkpoints()
    
    def _scan_existing_checkpoints(self):
        """Scan existing checkpoints in directory."""
        if not self.checkpoint_dir.exists():
            return
        
        for checkpoint_path in sorted(self.checkpoint_dir.glob("checkpoint-*")):
            if checkpoint_path.is_dir():
                self.checkpoints.append(checkpoint_path)
        
        logger.info(f"Found {len(self.checkpoints)} existing checkpoints")
    
    def save_checkpoint(
        self,
        model: NexusForCausalLM,
        step: int,
        epoch: Optional[int] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[CheckpointMetadata] = None,
        tags: Optional[List[str]] = None,
    ) -> Path:
        """Save model checkpoint."""
        
        # Create checkpoint directory
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        if metadata is None:
            metadata = CheckpointMetadata(
                model_name="nexus-7b",
                training_step=step,
                training_epoch=epoch or 0,
                global_step=step,
                train_loss=metrics.get("train_loss") if metrics else None,
                eval_loss=metrics.get("eval_loss") if metrics else None,
                perplexity=metrics.get("perplexity") if metrics else None,
                model_config=model.config.__dict__ if hasattr(model, 'config') else {},
                tags=tags or [],
            )
        
        # Save model weights
        model_file = self._save_model_weights(model, checkpoint_path)
        
        # Save optimizer state
        if self.save_optimizer and optimizer is not None:
            self._save_optimizer_state(optimizer, checkpoint_path)
        
        # Save scheduler state
        if self.save_scheduler and scheduler is not None:
            self._save_scheduler_state(scheduler, checkpoint_path)
        
        # Save metadata
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Compute checksum
        checksum = self._compute_checksum(checkpoint_path)
        metadata.checksum = checksum
        
        # Update metadata with checksum
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Track checkpoint
        self.checkpoints.append(checkpoint_path)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path
    
    def _save_model_weights(self, model: NexusForCausalLM, checkpoint_path: Path) -> Path:
        """Save model weights in safetensors or PyTorch format."""
        
        state_dict = model.state_dict()
        
        if self.save_safetensors:
            # Save in safetensors format
            model_file = checkpoint_path / "model.safetensors"
            save_file(state_dict, str(model_file))
            logger.debug(f"Saved model weights to {model_file} (safetensors)")
        else:
            # Save in PyTorch format
            model_file = checkpoint_path / "pytorch_model.bin"
            torch.save(state_dict, model_file)
            logger.debug(f"Saved model weights to {model_file} (PyTorch)")
        
        # Save config
        if hasattr(model, 'config'):
            config_file = checkpoint_path / "config.json"
            with open(config_file, 'w') as f:
                json.dump(model.config.__dict__, f, indent=2)
        
        return model_file
    
    def _save_optimizer_state(self, optimizer: Optimizer, checkpoint_path: Path):
        """Save optimizer state."""
        optimizer_file = checkpoint_path / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_file)
        logger.debug(f"Saved optimizer state to {optimizer_file}")
    
    def _save_scheduler_state(self, scheduler: Any, checkpoint_path: Path):
        """Save scheduler state."""
        scheduler_file = checkpoint_path / "scheduler.pt"
        torch.save(scheduler.state_dict() if hasattr(scheduler, 'state_dict') else scheduler, scheduler_file)
        logger.debug(f"Saved scheduler state to {scheduler_file}")
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: Optional[NexusForCausalLM] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[str] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load model checkpoint."""
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load metadata
        metadata_file = checkpoint_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = CheckpointMetadata.from_dict(json.load(f))
        else:
            metadata = None
        
        # Verify checksum
        if metadata and metadata.checksum:
            current_checksum = self._compute_checksum(checkpoint_path, exclude=["metadata.json"])
            if current_checksum != metadata.checksum:
                logger.warning("Checkpoint checksum mismatch! File may be corrupted.")
        
        # Load model weights
        if model is not None:
            self._load_model_weights(model, checkpoint_path, device, strict)
        
        # Load optimizer state
        if optimizer is not None:
            self._load_optimizer_state(optimizer, checkpoint_path)
        
        # Load scheduler state
        if scheduler is not None:
            self._load_scheduler_state(scheduler, checkpoint_path)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "metadata": metadata,
        }
    
    def _load_model_weights(
        self,
        model: NexusForCausalLM,
        checkpoint_path: Path,
        device: Optional[str],
        strict: bool,
    ):
        """Load model weights."""
        
        # Try safetensors first
        safetensors_file = checkpoint_path / "model.safetensors"
        pytorch_file = checkpoint_path / "pytorch_model.bin"
        
        if safetensors_file.exists() and SAFETENSORS_AVAILABLE:
            state_dict = load_file(str(safetensors_file))
            logger.debug(f"Loaded model weights from {safetensors_file} (safetensors)")
        elif pytorch_file.exists():
            state_dict = torch.load(pytorch_file, map_location=device or "cpu")
            logger.debug(f"Loaded model weights from {pytorch_file} (PyTorch)")
        else:
            raise FileNotFoundError(f"No model weights found in {checkpoint_path}")
        
        # Load state dict
        model.load_state_dict(state_dict, strict=strict)
    
    def _load_optimizer_state(self, optimizer: Optimizer, checkpoint_path: Path):
        """Load optimizer state."""
        optimizer_file = checkpoint_path / "optimizer.pt"
        
        if optimizer_file.exists():
            state_dict = torch.load(optimizer_file)
            optimizer.load_state_dict(state_dict)
            logger.debug(f"Loaded optimizer state from {optimizer_file}")
    
    def _load_scheduler_state(self, scheduler: Any, checkpoint_path: Path):
        """Load scheduler state."""
        scheduler_file = checkpoint_path / "scheduler.pt"
        
        if scheduler_file.exists():
            state_dict = torch.load(scheduler_file)
            if hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(state_dict)
            logger.debug(f"Loaded scheduler state from {scheduler_file}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        
        while len(self.checkpoints) > self.max_checkpoints:
            oldest_checkpoint = self.checkpoints.pop(0)
            
            if oldest_checkpoint.exists():
                shutil.rmtree(oldest_checkpoint)
                logger.info(f"Removed old checkpoint: {oldest_checkpoint}")
    
    def _compute_checksum(self, checkpoint_path: Path, exclude: Optional[List[str]] = None) -> str:
        """Compute checksum for checkpoint directory."""
        
        exclude = exclude or []
        hasher = hashlib.sha256()
        
        for file_path in sorted(checkpoint_path.rglob("*")):
            if file_path.is_file() and file_path.name not in exclude:
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        
        checkpoints_info = []
        
        for checkpoint_path in self.checkpoints:
            metadata_file = checkpoint_path / "metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"checkpoint_name": checkpoint_path.name}
            
            # Get file sizes
            total_size = sum(
                f.stat().st_size for f in checkpoint_path.rglob('*') if f.is_file()
            )
            
            checkpoints_info.append({
                "path": str(checkpoint_path),
                "name": checkpoint_path.name,
                "metadata": metadata,
                "size_bytes": total_size,
                "size_human": self._format_size(total_size),
            })
        
        return checkpoints_info
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format."""
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.2f} PB"
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the latest checkpoint path."""
        
        if not self.checkpoints:
            return None
        
        return self.checkpoints[-1]
    
    def get_best_checkpoint(self, metric: str = "eval_loss") -> Optional[Path]:
        """Get the best checkpoint based on a metric."""
        
        best_checkpoint = None
        best_metric_value = float('inf')
        
        for checkpoint_path in self.checkpoints:
            metadata_file = checkpoint_path / "metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                metric_value = metadata.get(metric)
                
                if metric_value is not None and metric_value < best_metric_value:
                    best_metric_value = metric_value
                    best_checkpoint = checkpoint_path
        
        return best_checkpoint
    
    def delete_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Delete a specific checkpoint."""
        
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
            
            if checkpoint_path in self.checkpoints:
                self.checkpoints.remove(checkpoint_path)
            
            logger.info(f"Deleted checkpoint: {checkpoint_path}")
    
    def export_to_huggingface(
        self,
        checkpoint_path: Union[str, Path],
        output_path: Union[str, Path],
        model_card: Optional[Dict[str, Any]] = None,
    ):
        """Export checkpoint to HuggingFace format."""
        
        checkpoint_path = Path(checkpoint_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint
        checkpoint = self.load_checkpoint(checkpoint_path)
        model = checkpoint["model"]
        metadata = checkpoint.get("metadata")
        
        # Save in HuggingFace format
        if self.save_safetensors:
            model_file = output_path / "model.safetensors"
            save_file(model.state_dict(), str(model_file))
        else:
            model_file = output_path / "pytorch_model.bin"
            torch.save(model.state_dict(), model_file)
        
        # Save config
        if hasattr(model, 'config'):
            config_file = output_path / "config.json"
            with open(config_file, 'w') as f:
                json.dump(model.config.__dict__, f, indent=2)
        
        # Create model card
        if model_card is None and metadata:
            model_card = {
                "model_description": f"Nexus-7B model checkpoint at step {metadata.training_step}",
                "training_info": {
                    "steps": metadata.training_step,
                    "train_loss": metadata.train_loss,
                    "eval_loss": metadata.eval_loss,
                },
                "tags": metadata.tags,
            }
        
        if model_card:
            card_file = output_path / "README.md"
            with open(card_file, 'w') as f:
                f.write(self._generate_model_card(model_card))
        
        logger.info(f"Exported checkpoint to HuggingFace format: {output_path}")
    
    def _generate_model_card(self, model_card: Dict[str, Any]) -> str:
        """Generate model card in markdown format."""
        
        lines = [
            "# Model Card",
            "",
            f"## Description",
            f"{model_card.get('model_description', 'Nexus-7B Language Model')}",
            "",
            "## Training Information",
        ]
        
        training_info = model_card.get('training_info', {})
        for key, value in training_info.items():
            lines.append(f"- **{key}**: {value}")
        
        lines.extend([
            "",
            "## Tags",
        ])
        
        tags = model_card.get('tags', [])
        if tags:
            lines.append(", ".join([f"`{tag}`" for tag in tags]))
        
        return "\n".join(lines)


class ShardedCheckpointManager(CheckpointManager):
    """Checkpoint manager with sharding support for large models."""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        shard_size: str = "5GB",
        **kwargs,
    ):
        super().__init__(checkpoint_dir, max_checkpoints, **kwargs)
        
        # Parse shard size
        self.shard_size = self._parse_size(shard_size)
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes."""
        
        units = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 ** 2,
            'GB': 1024 ** 3,
            'TB': 1024 ** 4,
        }
        
        size_str = size_str.upper()
        
        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                return int(float(size_str[:-len(unit)]) * multiplier)
        
        return int(size_str)
    
    def _save_model_weights(self, model: NexusForCausalLM, checkpoint_path: Path) -> Path:
        """Save model weights with sharding."""
        
        state_dict = model.state_dict()
        
        # Calculate shards
        total_size = sum(
            param.numel() * param.element_size()
            for param in state_dict.values()
        )
        
        num_shards = max(1, (total_size + self.shard_size - 1) // self.shard_size)
        
        if num_shards == 1:
            # Single shard - use parent class method
            return super()._save_model_weights(model, checkpoint_path)
        
        # Multiple shards
        shard_files = []
        
        # Group parameters into shards
        current_shard = {}
        current_shard_size = 0
        shard_index = 0
        
        for key, param in state_dict.items():
            param_size = param.numel() * param.element_size()
            
            if current_shard_size + param_size > self.shard_size and current_shard:
                # Save current shard
                shard_file = checkpoint_path / f"model-{shard_index:05d}-of-{num_shards:05d}.safetensors"
                save_file(current_shard, str(shard_file))
                shard_files.append(shard_file)
                
                # Start new shard
                current_shard = {key: param}
                current_shard_size = param_size
                shard_index += 1
            else:
                current_shard[key] = param
                current_shard_size += param_size
        
        # Save last shard
        if current_shard:
            shard_file = checkpoint_path / f"model-{shard_index:05d}-of-{num_shards:05d}.safetensors"
            save_file(current_shard, str(shard_file))
            shard_files.append(shard_file)
        
        # Save index file
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": {},
        }
        
        for i, shard_file in enumerate(shard_files):
            shard_dict = load_file(str(shard_file))
            for key in shard_dict.keys():
                index["weight_map"][key] = shard_file.name
        
        index_file = checkpoint_path / "model.safetensors.index.json"
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"Saved sharded model to {len(shard_files)} shards")
        
        return shard_files[0]
    
    def _load_model_weights(
        self,
        model: NexusForCausalLM,
        checkpoint_path: Path,
        device: Optional[str],
        strict: bool,
    ):
        """Load model weights with sharding support."""
        
        # Check for index file
        index_file = checkpoint_path / "model.safetensors.index.json"
        
        if index_file.exists():
            # Load sharded model
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            state_dict = {}
            loaded_shards = set()
            
            for key, shard_name in index["weight_map"].items():
                if shard_name not in loaded_shards:
                    shard_file = checkpoint_path / shard_name
                    shard_dict = load_file(str(shard_file))
                    state_dict.update(shard_dict)
                    loaded_shards.add(shard_name)
            
            model.load_state_dict(state_dict, strict=strict)
            logger.debug(f"Loaded sharded model from {len(loaded_shards)} shards")
        else:
            # Single file - use parent class method
            super()._load_model_weights(model, checkpoint_path, device, strict)


def save_pretrained(
    model: NexusForCausalLM,
    save_directory: Union[str, Path],
    save_safetensors: bool = True,
    **kwargs,
):
    """Convenience function to save model in pretrained format."""
    
    manager = CheckpointManager(
        checkpoint_dir=save_directory,
        save_safetensors=save_safetensors,
        **kwargs,
    )
    
    manager._save_model_weights(model, Path(save_directory))


def load_pretrained(
    model_class: type,
    model_path: Union[str, Path],
    config: Optional[NexusConfig] = None,
    device: Optional[str] = None,
    **kwargs,
) -> NexusForCausalLM:
    """Convenience function to load pretrained model."""
    
    manager = CheckpointManager(**kwargs)
    
    # Create model instance
    if config is None:
        # Load config from checkpoint
        config_file = Path(model_path) / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            config = NexusConfig(**config_dict)
        else:
            raise FileNotFoundError(f"Config not found: {config_file}")
    
    model = model_class(config)
    
    # Load weights
    manager._load_model_weights(model, Path(model_path), device, strict=True)
    
    return model
