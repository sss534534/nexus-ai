"""
Tests for Checkpoint Management Module
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from nexus_llm.model.checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    ShardedCheckpointManager,
    save_pretrained,
    load_pretrained,
)
from nexus_llm.model import NexusConfig, NexusForCausalLM


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata."""
    
    def test_default_metadata(self):
        """Test default metadata creation."""
        metadata = CheckpointMetadata(model_name="test-model")
        
        assert metadata.model_name == "test-model"
        assert metadata.model_version == "1.0.0"
        assert metadata.training_step == 0
        assert metadata.timestamp != ""
    
    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        metadata = CheckpointMetadata(
            model_name="test",
            training_step=100,
            train_loss=0.5,
        )
        
        data = metadata.to_dict()
        
        assert data["model_name"] == "test"
        assert data["training_step"] == 100
        assert data["train_loss"] == 0.5
    
    def test_metadata_from_dict(self):
        """Test metadata deserialization."""
        data = {
            "model_name": "test",
            "training_step": 100,
            "train_loss": 0.5,
        }
        
        metadata = CheckpointMetadata.from_dict(data)
        
        assert metadata.model_name == "test"
        assert metadata.training_step == 100


class TestCheckpointManager:
    """Tests for CheckpointManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def small_model(self):
        """Create small model for testing."""
        config = NexusConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=1000,
        )
        return NexusForCausalLM(config)
    
    def test_manager_creation(self, temp_dir):
        """Test checkpoint manager creation."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))
        
        assert manager.checkpoint_dir == temp_dir
        assert manager.max_checkpoints == 5
    
    def test_save_checkpoint(self, temp_dir, small_model):
        """Test saving checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))
        
        checkpoint_path = manager.save_checkpoint(
            model=small_model,
            step=100,
            epoch=1,
        )
        
        assert checkpoint_path.exists()
        assert (checkpoint_path / "metadata.json").exists()
    
    def test_load_checkpoint(self, temp_dir, small_model):
        """Test loading checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))
        
        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=small_model,
            step=100,
        )
        
        # Create new model
        config = NexusConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=1000,
        )
        new_model = NexusForCausalLM(config)
        
        # Load checkpoint
        result = manager.load_checkpoint(checkpoint_path, model=new_model)
        
        assert result["model"] is not None
        assert result["metadata"] is not None
    
    def test_checkpoint_cleanup(self, temp_dir, small_model):
        """Test old checkpoint cleanup."""
        manager = CheckpointManager(
            checkpoint_dir=str(temp_dir),
            max_checkpoints=2,
        )
        
        # Save multiple checkpoints
        for i in range(4):
            manager.save_checkpoint(
                model=small_model,
                step=i * 100,
            )
        
        # Should only keep 2 checkpoints
        assert len(manager.checkpoints) == 2
    
    def test_list_checkpoints(self, temp_dir, small_model):
        """Test listing checkpoints."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))
        
        # Save checkpoints
        manager.save_checkpoint(model=small_model, step=100)
        manager.save_checkpoint(model=small_model, step=200)
        
        # List checkpoints
        checkpoints = manager.list_checkpoints()
        
        assert len(checkpoints) == 2
        assert all("path" in cp for cp in checkpoints)
        assert all("metadata" in cp for cp in checkpoints)
    
    def test_get_latest_checkpoint(self, temp_dir, small_model):
        """Test getting latest checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))
        
        # Save checkpoints
        manager.save_checkpoint(model=small_model, step=100)
        latest = manager.save_checkpoint(model=small_model, step=200)
        
        # Get latest
        result = manager.get_latest_checkpoint()
        
        assert result == latest
    
    def test_get_best_checkpoint(self, temp_dir, small_model):
        """Test getting best checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))
        
        # Save checkpoints with different losses
        manager.save_checkpoint(
            model=small_model,
            step=100,
            metrics={"eval_loss": 0.5},
        )
        best = manager.save_checkpoint(
            model=small_model,
            step=200,
            metrics={"eval_loss": 0.3},
        )
        manager.save_checkpoint(
            model=small_model,
            step=300,
            metrics={"eval_loss": 0.4},
        )
        
        # Get best
        result = manager.get_best_checkpoint("eval_loss")
        
        assert result == best
    
    def test_delete_checkpoint(self, temp_dir, small_model):
        """Test deleting checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))
        
        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=small_model,
            step=100,
        )
        
        # Delete
        manager.delete_checkpoint(checkpoint_path)
        
        assert not checkpoint_path.exists()
        assert checkpoint_path not in manager.checkpoints


class TestShardedCheckpointManager:
    """Tests for ShardedCheckpointManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def large_model(self):
        """Create larger model for sharding tests."""
        config = NexusConfig(
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=8,
            vocab_size=50000,
        )
        return NexusForCausalLM(config)
    
    def test_sharded_save(self, temp_dir, large_model):
        """Test saving sharded checkpoint."""
        manager = ShardedCheckpointManager(
            checkpoint_dir=str(temp_dir),
            shard_size="1MB",
        )
        
        checkpoint_path = manager.save_checkpoint(
            model=large_model,
            step=100,
        )
        
        # Should create index file
        assert (checkpoint_path / "model.safetensors.index.json").exists()
    
    def test_sharded_load(self, temp_dir, large_model):
        """Test loading sharded checkpoint."""
        manager = ShardedCheckpointManager(
            checkpoint_dir=str(temp_dir),
            shard_size="1MB",
        )
        
        # Save
        checkpoint_path = manager.save_checkpoint(
            model=large_model,
            step=100,
        )
        
        # Create new model and load
        config = NexusConfig(
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=8,
            vocab_size=50000,
        )
        new_model = NexusForCausalLM(config)
        
        result = manager.load_checkpoint(checkpoint_path, model=new_model)
        
        assert result["model"] is not None


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        config = NexusConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=1000,
        )
        return NexusForCausalLM(config)
    
    def test_save_pretrained(self, temp_dir, model):
        """Test save_pretrained function."""
        save_pretrained(model, str(temp_dir / "model"))
        
        assert (temp_dir / "model").exists()
        assert (temp_dir / "model" / "config.json").exists()
    
    def test_load_pretrained(self, temp_dir, model):
        """Test load_pretrained function."""
        # Save first
        save_pretrained(model, str(temp_dir / "model"))
        
        # Load
        loaded_model = load_pretrained(
            NexusForCausalLM,
            str(temp_dir / "model"),
        )
        
        assert loaded_model is not None
        assert isinstance(loaded_model, NexusForCausalLM)


class TestCheckpointChecksum:
    """Tests for checkpoint checksum verification."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        config = NexusConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=1000,
        )
        return NexusForCausalLM(config)
    
    def test_checksum_computation(self, temp_dir, model):
        """Test checksum computation."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))
        
        checkpoint_path = manager.save_checkpoint(
            model=model,
            step=100,
        )
        
        # Check metadata has checksum
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert metadata["checksum"] is not None
        assert len(metadata["checksum"]) == 64  # SHA-256 hex


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
