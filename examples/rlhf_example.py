"""
RLHF Training Example
Demonstrates how to use the RLHF module for training with human feedback
"""
import torch
from nexus_llm.rlhf import (
    RLHFConfig,
    RewardModel,
    RewardModelTrainer,
    PPOTrainer,
    create_rlhf_pipeline
)


def example_reward_model_training():
    """Example: Training a Reward Model"""
    print("=" * 50)
    print("Example: Reward Model Training")
    print("=" * 50)
    
    # Configure the reward model
    config = RLHFConfig(
        vocab_size=32000,
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1024,
        reward_model_hidden_size=512,
        reward_learning_rate=1e-4,
        reward_num_epochs=3,
        reward_batch_size=8
    )
    
    # Create trainer
    trainer = RewardModelTrainer(config)
    
    # Simulate preference data (preferred and rejected responses)
    batch_size = 8
    seq_len = 128
    
    preferred_inputs = torch.randint(0, 32000, (batch_size, seq_len))
    rejected_inputs = torch.randint(0, 32000, (batch_size, seq_len))
    preferred_masks = torch.ones(batch_size, seq_len)
    rejected_masks = torch.ones(batch_size, seq_len)
    
    print(f"Training on {batch_size} preference pairs...")
    print(f"Sequence length: {seq_len}")
    
    # Train for a few steps
    for step in range(3):
        loss = trainer.train_step(
            preferred_inputs,
            preferred_masks,
            rejected_inputs,
            rejected_masks
        )
        print(f"Step {step + 1}: Loss = {loss:.4f}")
    
    print("Reward model training complete!")
    return trainer


def example_ppo_training():
    """Example: PPO Training with RLHF"""
    print("\n" + "=" * 50)
    print("Example: PPO Training")
    print("=" * 50)
    
    # Configure PPO
    config = RLHFConfig(
        vocab_size=32000,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        actor_hidden_size=256,
        critic_hidden_size=256,
        ppo_clip_eps=0.2,
        ppo_entropy_coef=0.01,
        ppo_learning_rate=1e-5,
        ppo_init_kl_coef=0.02,
        gamma=0.99,
        lam=0.95
    )
    
    # Create PPO trainer
    trainer = PPOTrainer(config)
    
    # Simulate prompts
    batch_size = 4
    seq_len = 64
    
    prompts = torch.randint(0, 32000, (batch_size, seq_len))
    prompt_masks = torch.ones(batch_size, seq_len)
    
    print(f"Training PPO with batch size {batch_size}...")
    
    # PPO training steps
    for step in range(3):
        # Simulate rewards (in real use, these come from human feedback)
        rewards = torch.randn(batch_size)
        
        stats = trainer.train_step(
            prompts,
            prompt_masks,
            rewards
        )
        
        print(f"Step {step + 1}: "
              f"Policy Loss = {stats.get('policy_loss', 0):.4f}, "
              f"Value Loss = {stats.get('value_loss', 0):.4f}, "
              f"Entropy = {stats.get('entropy', 0):.4f}")
    
    print("PPO training complete!")
    return trainer


def example_full_pipeline():
    """Example: Full RLHF Pipeline"""
    print("\n" + "=" * 50)
    print("Example: Full RLHF Pipeline")
    print("=" * 50)
    
    config = RLHFConfig(
        vocab_size=32000,
        hidden_size=256,
        num_attention_heads=4,
        intermediate_size=512,
        ppo_clip_eps=0.2,
        ppo_entropy_coef=0.01
    )
    
    # Create full pipeline
    pipeline = create_rlhf_pipeline(config)
    
    print("Pipeline components:")
    print(f"  - Reward Model: {type(pipeline.reward_model).__name__}")
    print(f"  - PPO Trainer: {type(pipeline.ppo_trainer).__name__}")
    print(f"  - Actor: {type(pipeline.ppo_trainer.actor).__name__}")
    print(f"  - Critic: {type(pipeline.ppo_trainer.critic).__name__}")
    
    print("\nRLHF Pipeline ready for training!")
    return pipeline


def example_kl_penalty_types():
    """Example: Different KL Penalty Types"""
    print("\n" + "=" * 50)
    print("Example: KL Penalty Types")
    print("=" * 50)
    
    for kl_penalty in ["kl", "kl-linear", "kl-log"]:
        config = RLHFConfig(
            vocab_size=32000,
            hidden_size=128,
            num_attention_heads=2,
            intermediate_size=256,
            kl_penalty=kl_penalty
        )
        
        trainer = PPOTrainer(config)
        print(f"KL Penalty: {kl_penalty}")
        print(f"  - Type: {trainer.kl_penalty_type}")
        print(f"  - Init Coef: {trainer.kl_coef}")


if __name__ == "__main__":
    # Run all examples
    reward_trainer = example_reward_model_training()
    ppo_trainer = example_ppo_training()
    pipeline = example_full_pipeline()
    example_kl_penalty_types()
    
    print("\n" + "=" * 50)
    print("All RLHF examples completed!")
    print("=" * 50)
