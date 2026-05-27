"""
Example: Training Pipeline
"""

import torch
from pathlib import Path

from nexus_llm import NexusConfig, NexusForCausalLM
from nexus_llm.data import NexusTokenizer, DataProcessor, DataConfig
from nexus_llm.training import Trainer, TrainingConfig, create_trainer


def prepare_data():
    """Prepare training data."""
    
    # Create sample data directory
    data_dir = Path("data/sample")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample training data
    sample_data = [
        {"text": "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。"},
        {"text": "自然语言处理是人工智能的一个重要领域，专注于让计算机理解和生成人类语言。"},
        {"text": "Transformer架构通过自注意力机制实现了并行处理序列数据的能力。"},
        {"text": "大语言模型通过在海量文本数据上训练，获得了强大的语言理解和生成能力。"},
    ]
    
    # Write training data
    import json
    with open(data_dir / "train.jsonl", "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    # Write validation data
    with open(data_dir / "valid.jsonl", "w") as f:
        for item in sample_data[:2]:
            f.write(json.dumps(item) + "\n")
    
    print(f"Sample data created in {data_dir}")
    
    return str(data_dir / "train.jsonl"), str(data_dir / "valid.jsonl")


def train_example():
    """Complete training example."""
    
    print("=" * 50)
    print("Nexus-7B Training Example")
    print("=" * 50)
    
    # Step 1: Prepare data
    print("\n1. Preparing data...")
    train_file, valid_file = prepare_data()
    
    # Step 2: Create configurations
    print("\n2. Creating configurations...")
    
    # Model configuration (smaller for demo)
    model_config = NexusConfig(
        hidden_size=512,  # Smaller for demo
        intermediate_size=1408,
        num_attention_heads=8,
        num_hidden_layers=8,
        vocab_size=10000,
        max_position_embeddings=512,
        use_flash_attention=False,  # Disable for demo
    )
    
    # Data configuration
    data_config = DataConfig(
        train_file=train_file,
        valid_file=valid_file,
        max_seq_length=512,
        batch_size=2,
        num_workers=0,
    )
    
    # Training configuration
    training_config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=2,
        max_steps=100,  # Small for demo
        save_steps=50,
        eval_steps=25,
        log_steps=10,
        output_dir="outputs/demo",
        checkpoint_dir="outputs/demo/checkpoints",
        use_wandb=False,  # Disable for demo
        use_tensorboard=False,
        precision="fp32",  # Use fp32 for demo
    )
    
    # Step 3: Create model
    print("\n3. Creating model...")
    model = NexusForCausalLM(model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"   Device: {device}")
    
    # Step 4: Create tokenizer (mock for demo)
    print("\n4. Setting up tokenizer...")
    # Note: In real training, you need to train or load a proper tokenizer
    # This is a simplified example
    
    # Step 5: Create trainer
    print("\n5. Creating trainer...")
    # Note: Full training requires proper data pipeline
    # This demonstrates the setup
    
    print("\n6. Training setup complete!")
    print("\nTo run actual training:")
    print("   nexus-train --model-config configs/model_config.yaml")
    print("   --training-config configs/training_config.yaml")
    print("   --data-config configs/data_config.yaml")


def distributed_training_example():
    """Example of distributed training setup."""
    
    print("\n" + "=" * 50)
    print("Distributed Training Example")
    print("=" * 50)
    
    print("\nFor distributed training, use:")
    print("\n   # 8-GPU single node")
    print("   torchrun --nproc_per_node=8 -m nexus_llm.cli train \\")
    print("       --model-config configs/model_config.yaml \\")
    print("       --distributed")
    
    print("\n   # Multi-node (2 nodes, 8 GPUs each)")
    print("   # Node 0 (master):")
    print("   torchrun --nnodes=2 --nproc_per_node=8 \\")
    print("       --master_addr=10.0.0.1 --master_port=29500 \\")
    print("       -m nexus_llm.cli train --distributed")
    
    print("\n   # Node 1:")
    print("   torchrun --nnodes=2 --nproc_per_node=8 \\")
    print("       --master_addr=10.0.0.1 --master_port=29500 \\")
    print("       -m nexus_llm.cli train --distributed")


def lora_finetuning_example():
    """Example of LoRA fine-tuning."""
    
    print("\n" + "=" * 50)
    print("LoRA Fine-tuning Example")
    print("=" * 50)
    
    print("\nLoRA configuration:")
    
    lora_config = {
        "r": 16,
        "alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "dropout": 0.05,
    }
    
    print(f"   Rank (r): {lora_config['r']}")
    print(f"   Alpha: {lora_config['alpha']}")
    print(f"   Target modules: {lora_config['target_modules']}")
    
    print("\nPython API:")
    print("""
    from nexus_llm.training import FineTuningTrainer
    
    trainer = FineTuningTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        config=training_config,
        lora_config=lora_config,
    )
    trainer.train()
    """)
    
    print("\nCommand line:")
    print("   nexus-train --lora-config configs/lora_config.yaml")


if __name__ == "__main__":
    train_example()
    distributed_training_example()
    lora_finetuning_example()