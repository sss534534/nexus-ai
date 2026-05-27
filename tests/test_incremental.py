"""
Incremental Learning Module Tests
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from nexus_llm.incremental import (
    IncrementalConfig,
    ExperienceReplayBuffer,
    EWCRegularizer,
    AdapterModule,
    IncrementalTrainer,
    create_incremental_trainer
)


class TestIncrementalConfig:
    """Test Incremental Learning Configuration"""
    
    def test_default_config(self):
        config = IncrementalConfig()
        assert config.strategy == "adapter"
        assert config.memory_size == 1000
        assert config.ewc_lambda == 1000.0
        assert config.adapter_dim == 64
        assert config.fisher_sample_size == 100
    
    def test_custom_config(self):
        config = IncrementalConfig(
            strategy="regularization",
            memory_size=2000,
            ewc_lambda=5000.0,
            adapter_dim=128
        )
        assert config.strategy == "regularization"
        assert config.memory_size == 2000
        assert config.ewc_lambda == 5000.0
        assert config.adapter_dim == 128
    
    def test_adapter_config(self):
        config = IncrementalConfig(
            strategy="adapter",
            adapter_dim=64,
            adapter_dropout=0.1
        )
        assert config.adapter_dim == 64
        assert config.adapter_dropout == 0.1


class MockModel(nn.Module):
    """Mock model for testing"""
    def __init__(self, hidden_size=256):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class TestExperienceReplayBuffer:
    """Test Experience Replay Buffer"""
    
    @pytest.fixture
    def buffer(self):
        return ExperienceReplayBuffer(max_size=100)
    
    @pytest.fixture
    def sample_data(self):
        return {
            "input_ids": torch.randint(0, 1000, (10,)),
            "labels": torch.randint(0, 10, (10,)),
            "attention_mask": torch.ones(10)
        }
    
    def test_buffer_initialization(self, buffer):
        assert buffer.max_size == 100
        assert len(buffer) == 0
        assert buffer.is_empty()
    
    def test_add_experience(self, buffer, sample_data):
        buffer.add_experience(sample_data)
        
        assert len(buffer) == 1
        assert not buffer.is_empty()
    
    def test_buffer_overflow(self, buffer):
        # Add more than max_size items
        for i in range(150):
            buffer.add_experience({
                "input_ids": torch.tensor([i]),
                "labels": torch.tensor([i % 10]),
                "attention_mask": torch.tensor([1])
            })
        
        # Should not exceed max_size
        assert len(buffer) == 100
    
    def test_sample_batch(self, buffer):
        # Add some experiences
        for i in range(20):
            buffer.add_experience({
                "input_ids": torch.tensor([i] * 10),
                "labels": torch.tensor([i % 10]),
                "attention_mask": torch.tensor([1] * 10)
            })
        
        batch = buffer.sample_batch(batch_size=4)
        
        assert len(batch) == 4
        assert "input_ids" in batch[0]
        assert "labels" in batch[0]
    
    def test_buffer_sample_with_replacement(self, buffer):
        # With small buffer, should sample with replacement
        for i in range(5):
            buffer.add_experience({
                "input_ids": torch.tensor([i]),
                "labels": torch.tensor([i]),
                "attention_mask": torch.tensor([1])
            })
        
        # Sample more than buffer size
        samples = buffer.sample_batch(batch_size=10, replacement=True)
        assert len(samples) == 10
    
    def test_clear_buffer(self, buffer, sample_data):
        buffer.add_experience(sample_data)
        buffer.add_experience(sample_data)
        
        assert len(buffer) == 2
        
        buffer.clear()
        
        assert len(buffer) == 0
        assert buffer.is_empty()
    
    def test_save_load_buffer(self, buffer, sample_data):
        buffer.add_experience(sample_data)
        
        # Save state
        state = buffer.state_dict()
        
        # Create new buffer and load
        new_buffer = ExperienceReplayBuffer(max_size=100)
        new_buffer.load_state_dict(state)
        
        assert len(new_buffer) == len(buffer)


class TestEWCRegularizer:
    """Test EWC Regularizer"""
    
    @pytest.fixture
    def model(self):
        return MockModel(hidden_size=128)
    
    @pytest.fixture
    def regularizer(self, model):
        config = IncrementalConfig(
            strategy="regularization",
            ewc_lambda=1000.0
        )
        return EWCRegularizer(model, config)
    
    def test_regularizer_initialization(self, regularizer, model):
        assert regularizer.model is not None
        assert regularizer.ewc_lambda == 1000.0
        assert regularizer.fisher_dict is not None
        assert regularizer.params_dict is not None
    
    def test_compute_fisher(self, regularizer):
        # Mock input data
        inputs = torch.randn(10, 128)
        labels = torch.randint(0, 10, (10,))
        
        regularizer.compute_fisher(inputs, sample_size=5)
        
        # Fisher information should be computed
        assert len(regularizer.fisher_dict) > 0
    
    def test_penalty_computation(self, regularizer):
        # Set some stored parameters and fisher
        for name, param in regularizer.model.named_parameters():
            regularizer.params_dict[name] = param.data.clone()
            regularizer.fisher_dict[name] = torch.ones_like(param.data)
        
        # Compute penalty
        penalty = regularizer.penalty()
        
        assert isinstance(penalty.item(), float)
        assert penalty.item() >= 0
    
    def test_ewc_loss(self, regularizer):
        # Set up mock fisher and params
        for name, param in regularizer.model.named_parameters():
            regularizer.params_dict[name] = torch.randn_like(param)
            regularizer.fisher_dict[name] = torch.rand_like(param)
        
        # Compute EWC loss component
        loss = regularizer.ewc_loss()
        
        assert isinstance(loss.item(), float)
    
    def test_save_load_ewc(self, regularizer):
        # Compute some fisher information
        inputs = torch.randn(5, 128)
        labels = torch.randint(0, 10, (5,))
        regularizer.compute_fisher(inputs, sample_size=3)
        
        # Save state
        state = regularizer.state_dict()
        
        # Create new regularizer and load
        new_regularizer = EWCRegularizer(MockModel(128), IncrementalConfig())
        new_regularizer.load_state_dict(state)
        
        assert len(new_regularizer.fisher_dict) == len(regularizer.fisher_dict)


class TestAdapterModule:
    """Test Adapter Module"""
    
    @pytest.fixture
    def adapter(self):
        config = IncrementalConfig(
            strategy="adapter",
            adapter_dim=32,
            adapter_dropout=0.1
        )
        model = MockModel(hidden_size=128)
        return AdapterModule(model, config), model
    
    def test_adapter_initialization(self, adapter):
        adapter_module, original_model = adapter
        
        assert adapter_module.adapter is not None
        assert adapter_module.bottleneck_dim == 32
        assert adapter_module.dropout.p == 0.1
    
    def test_adapter_forward(self, adapter):
        adapter_module, _ = adapter
        
        x = torch.randn(4, 128)
        output = adapter_module.forward(x)
        
        assert output.shape == (4, 128)
    
    def test_freeze_base_model(self, adapter):
        adapter_module, _ = adapter
        
        adapter_module.freeze_base_model()
        
        # Base model parameters should be frozen
        for name, param in adapter_module.model.named_parameters():
            if "adapter" not in name:
                assert not param.requires_grad or param.grad is None
    
    def test_train_adapter_only(self, adapter):
        adapter_module, _ = adapter
        
        adapter_module.freeze_base_model()
        
        trainable_params = sum(p.numel() for p in adapter_module.model.parameters() if p.requires_grad)
        
        # Only adapter parameters should be trainable
        adapter_params = sum(p.numel() for p in adapter_module.adapter.parameters())
        
        assert trainable_params == adapter_params
    
    def test_adapter_scale(self, adapter):
        adapter_module, _ = adapter
        
        x = torch.randn(4, 128)
        
        # Test with different scales
        for scale in [0.5, 1.0, 2.0]:
            adapter_module.scale = scale
            output = adapter_module.forward(x)
            assert output.shape == (4, 128)


class TestIncrementalTrainer:
    """Test Incremental Trainer"""
    
    @pytest.fixture
    def trainer_config(self):
        return IncrementalConfig(
            strategy="adapter",
            memory_size=100,
            adapter_dim=32,
            learning_rate=1e-3
        )
    
    @pytest.fixture
    def trainer(self, trainer_config):
        model = MockModel(hidden_size=128)
        return IncrementalTrainer(model, trainer_config)
    
    def test_trainer_initialization(self, trainer):
        assert trainer.model is not None
        assert trainer.config is not None
        assert trainer.strategy == "adapter"
    
    def test_strategy_selection(self, trainer_config):
        # Test different strategies
        for strategy in ["adapter", "replay", "regularization", "elastic"]:
            trainer_config.strategy = strategy
            trainer = IncrementalTrainer(MockModel(128), trainer_config)
            assert trainer.strategy == strategy
    
    def test_add_task(self, trainer):
        task_id = trainer.add_task("task1")
        
        assert task_id == 0
        assert len(trainer.task_configs) == 1
    
    def test_task_training_step(self, trainer):
        trainer.add_task("task1")
        
        # Mock training data
        inputs = torch.randn(4, 128)
        labels = torch.randint(0, 10, (4,))
        
        loss = trainer.train_step(inputs, labels)
        
        assert isinstance(loss.item(), float)
    
    def test_compute_memory_loss(self, trainer):
        # Add experiences to memory
        for i in range(10):
            trainer.memory.add_experience({
                "input_ids": torch.tensor([i]),
                "labels": torch.tensor([i % 5]),
                "attention_mask": torch.tensor([1])
            })
        
        # Compute memory loss
        if len(trainer.memory) > 0:
            loss = trainer.compute_memory_loss(batch_size=4)
            assert isinstance(loss.item(), float)
    
    def test_save_load_trainer(self, trainer, tmp_path):
        trainer.add_task("task1")
        trainer.add_task("task2")
        
        # Save state
        save_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(save_path))
        
        assert save_path.exists()
        
        # Load into new trainer
        new_trainer = IncrementalTrainer(MockModel(128), trainer.config)
        new_trainer.load_checkpoint(str(save_path))
        
        assert len(new_trainer.task_configs) == len(trainer.task_configs)


class TestIncrementalPipeline:
    """Test Incremental Learning Pipeline Factory"""
    
    def test_create_trainer(self):
        config = IncrementalConfig(
            strategy="adapter",
            memory_size=100
        )
        
        trainer = create_incremental_trainer(config)
        
        assert trainer is not None
        assert trainer.strategy == "adapter"
    
    def test_create_with_different_strategies(self):
        for strategy in ["adapter", "replay", "regularization"]:
            config = IncrementalConfig(strategy=strategy)
            trainer = create_incremental_trainer(config)
            assert trainer.strategy == strategy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
