"""
Incremental Learning Example
Demonstrates how to use the incremental learning module
"""
import torch
import torch.nn as nn
from nexus_llm.incremental import (
    IncrementalConfig,
    IncrementalTrainer,
    ExperienceReplayBuffer,
    EWCRegularizer,
    AdapterModule,
    create_incremental_trainer
)


# Simple Model for Testing
class SimpleLLM(nn.Module):
    """Simple LLM-like model"""
    def __init__(self, vocab_size=1000, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ),
            num_layers=2
        )
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)


def example_experience_replay():
    """Example: Experience Replay Buffer"""
    print("=" * 50)
    print("Example: Experience Replay Buffer")
    print("=" * 50)
    
    # Create buffer
    buffer = ExperienceReplayBuffer(max_size=100)
    
    print(f"Buffer capacity: {buffer.max_size}")
    print(f"Initial size: {len(buffer)}")
    
    # Add experiences
    print("\nAdding experiences...")
    for i in range(20):
        experience = {
            "input_ids": torch.randint(0, 1000, (16,)),
            "labels": torch.tensor([i % 10]),
            "attention_mask": torch.ones(16),
            "task_id": i % 3  # 3 different tasks
        }
        buffer.add_experience(experience)
    
    print(f"After adding 20 experiences: {len(buffer)}")
    
    # Sample batch
    batch = buffer.sample_batch(batch_size=5)
    print(f"\nSampled batch size: {len(batch)}")
    print(f"Batch keys: {list(batch[0].keys())}")
    
    # Sample more
    batch_large = buffer.sample_batch(batch_size=50, replacement=True)
    print(f"Sampled batch with replacement (size 50): {len(batch_large)}")
    
    # Save and load
    state = buffer.state_dict()
    new_buffer = ExperienceReplayBuffer(max_size=100)
    new_buffer.load_state_dict(state)
    print(f"\nBuffer state saved and loaded: {len(new_buffer)} experiences")
    
    return buffer


def example_ewc_regularization():
    """Example: EWC Regularization"""
    print("\n" + "=" * 50)
    print("Example: EWC Regularization")
    print("=" * 50)
    
    # Create model
    model = SimpleLLM(vocab_size=500, hidden_size=64)
    
    # Configure EWC
    config = IncrementalConfig(
        strategy="regularization",
        ewc_lambda=1000.0,  # EWC regularization strength
        fisher_sample_size=50
    )
    
    # Create regularizer
    regularizer = EWCRegularizer(model, config)
    
    print(f"Strategy: {regularizer.ewc_lambda}")
    print(f"Lambda: {regularizer.ewc_lambda}")
    
    # Compute fisher information
    print("\nComputing Fisher information...")
    sample_inputs = torch.randint(0, 500, (10, 16))
    sample_labels = torch.randint(0, 500, (10, 16))
    
    regularizer.compute_fisher(
        sample_inputs,
        labels=sample_labels,
        sample_size=5
    )
    
    print(f"Fisher dict size: {len(regularizer.fisher_dict)}")
    print(f"Params dict size: {len(regularizer.params_dict)}")
    
    # Compute penalty
    penalty = regularizer.penalty()
    print(f"\nEWC Penalty: {penalty.item():.4f}")
    
    # EWC loss
    ewc_loss = regularizer.ewc_loss()
    print(f"EWC Loss: {ewc_loss.item():.4f}")
    
    # Save and load
    state = regularizer.state_dict()
    new_regularizer = EWCRegularizer(SimpleLLM(500, 64), config)
    new_regularizer.load_state_dict(state)
    print("\nEWC state saved and loaded successfully")
    
    return regularizer


def example_adapter_training():
    """Example: Adapter-based Training"""
    print("\n" + "=" * 50)
    print("Example: Adapter Training")
    print("=" * 50)
    
    # Create model
    model = SimpleLLM(vocab_size=500, hidden_size=64)
    
    # Configure adapter
    config = IncrementalConfig(
        strategy="adapter",
        adapter_dim=32,  # Bottleneck dimension
        adapter_dropout=0.1
    )
    
    # Create adapter module
    adapter_module = AdapterModule(model, config)
    
    print(f"Adapter strategy: {config.strategy}")
    print(f"Adapter dim: {adapter_module.bottleneck_dim}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    adapter_params = sum(p.numel() for p in adapter_module.adapter.parameters())
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Adapter parameters: {adapter_params:,}")
    print(f"Parameter ratio: {adapter_params / total_params:.2%}")
    
    # Freeze base model
    print("\nFreezing base model...")
    adapter_module.freeze_base_model()
    
    # Count trainable parameters
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable parameters after freeze: {trainable_params:,}")
    
    # Forward pass
    x = torch.randint(0, 500, (2, 16))
    output = adapter_module.forward(x)
    print(f"\nOutput shape: {output.shape}")
    
    # Test with different scales
    print("\nTesting different adapter scales:")
    for scale in [0.5, 1.0, 2.0]:
        adapter_module.scale = scale
        output = adapter_module.forward(x)
        print(f"  Scale {scale}: output mean = {output.mean().item():.4f}")
    
    return adapter_module


def example_incremental_trainer():
    """Example: Incremental Trainer"""
    print("\n" + "=" * 50)
    print("Example: Incremental Trainer")
    print("=" * 50)
    
    # Create model
    model = SimpleLLM(vocab_size=500, hidden_size=64)
    
    # Configure trainer
    config = IncrementalConfig(
        strategy="adapter",
        memory_size=50,
        adapter_dim=32,
        learning_rate=1e-3
    )
    
    # Create trainer
    trainer = IncrementalTrainer(model, config)
    
    print(f"Strategy: {trainer.strategy}")
    print(f"Memory size: {trainer.config.memory_size}")
    print(f"Learning rate: {trainer.config.learning_rate}")
    
    # Add tasks
    print("\nAdding tasks...")
    task1_id = trainer.add_task("Task 1: General QA")
    task2_id = trainer.add_task("Task 2: Code Generation")
    task3_id = trainer.add_task("Task 3: Summarization")
    
    print(f"Added {len(trainer.task_configs)} tasks:")
    for task_id, task_config in trainer.task_configs.items():
        print(f"  Task {task_id}: {task_config['name']}")
    
    # Training step
    print("\nTraining step...")
    inputs = torch.randint(0, 500, (4, 16))
    labels = torch.randint(0, 500, (4, 16))
    
    loss = trainer.train_step(inputs, labels)
    print(f"Training loss: {loss.item():.4f}")
    
    # Add to memory
    for _ in range(10):
        trainer.memory.add_experience({
            "input_ids": torch.randint(0, 500, (16,)),
            "labels": torch.randint(0, 500, (16,)),
            "attention_mask": torch.ones(16)
        })
    
    print(f"\nMemory size: {len(trainer.memory)}")
    
    # Compute memory loss
    if len(trainer.memory) > 0:
        memory_loss = trainer.compute_memory_loss(batch_size=4)
        print(f"Memory loss: {memory_loss.item():.4f}")
    
    return trainer


def example_multi_strategy_comparison():
    """Example: Compare Different Strategies"""
    print("\n" + "=" * 50)
    print("Example: Strategy Comparison")
    print("=" * 50)
    
    strategies = ["adapter", "replay", "regularization"]
    
    for strategy in strategies:
        model = SimpleLLM(vocab_size=500, hidden_size=64)
        
        config = IncrementalConfig(
            strategy=strategy,
            memory_size=50 if strategy == "replay" else 0,
            ewc_lambda=1000.0 if strategy == "regularization" else 0,
            adapter_dim=32 if strategy == "adapter" else 0
        )
        
        trainer = IncrementalTrainer(model, config)
        
        print(f"\n{strategy.capitalize():15s} Strategy:")
        print(f"  Memory: {trainer.config.memory_size}")
        print(f"  EWC Lambda: {trainer.config.ewc_lambda}")
        print(f"  Adapter Dim: {trainer.config.adapter_dim}")


def example_pipeline():
    """Example: Incremental Pipeline Factory"""
    print("\n" + "=" * 50)
    print("Example: Incremental Pipeline")
    print("=" * 50)
    
    # Create pipeline
    config = IncrementalConfig(
        strategy="adapter",
        memory_size=100,
        adapter_dim=64
    )
    
    trainer = create_incremental_trainer(config)
    
    print(f"Pipeline created with strategy: {trainer.strategy}")
    print(f"Memory capacity: {trainer.memory.max_size}")
    
    # Simulate task learning
    trainer.add_task("Domain A: Medical")
    trainer.add_task("Domain B: Legal")
    
    print(f"Tasks registered: {len(trainer.task_configs)}")
    
    return trainer


def example_catastrophic_forgetting_prevention():
    """Example: How to Prevent Catastrophic Forgetting"""
    print("\n" + "=" * 50)
    print("Example: Preventing Catastrophic Forgetting")
    print("=" * 50)
    
    # Without EWC - parameters change freely
    model_base = SimpleLLM(vocab_size=500, hidden_size=64)
    
    # With EWC - parameters constrained
    config = IncrementalConfig(
        strategy="regularization",
        ewc_lambda=5000.0
    )
    model_ewc = SimpleLLM(vocab_size=500, hidden_size=64)
    
    # Store initial parameters for EWC model
    regularizer = EWCRegularizer(model_ewc, config)
    
    # Compute fisher on task A data
    task_a_data = torch.randint(0, 500, (20, 16))
    task_a_labels = torch.randint(0, 500, (20, 16))
    regularizer.compute_fisher(task_a_data, labels=task_a_labels, sample_size=10)
    
    # Store params after learning task A
    for name, param in model_ewc.named_parameters():
        regularizer.params_dict[name] = param.data.clone()
    
    print("Task A learned. Fisher information computed.")
    print(f"  Fisher dict entries: {len(regularizer.fisher_dict)}")
    
    # Simulate learning task B (would normally hurt task A performance)
    print("\nSimulating task B learning...")
    print("  Without EWC: Parameters freely updated (forgetting risk)")
    print("  With EWC: Additional penalty constrains parameter changes")
    
    ewc_penalty = regularizer.penalty()
    print(f"\nEWC penalty before task B: {ewc_penalty.item():.4f}")
    
    # The penalty ensures parameters don't drift too far from task A optimal
    print("\nEWC prevents catastrophic forgetting by penalizing parameter changes")
    print("that would hurt performance on previously learned tasks.")


if __name__ == "__main__":
    # Run all examples
    buffer = example_experience_replay()
    regularizer = example_ewc_regularization()
    adapter = example_adapter_training()
    trainer = example_incremental_trainer()
    example_multi_strategy_comparison()
    pipeline = example_pipeline()
    example_catastrophic_forgetting_prevention()
    
    print("\n" + "=" * 50)
    print("All Incremental Learning examples completed!")
    print("=" * 50)
