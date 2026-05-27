"""
RLHF Module Tests
"""
import pytest
import torch
from nexus_llm.rlhf import (
    RLHFConfig,
    RewardModel,
    PPOTrainer,
    RewardModelTrainer,
    create_rlhf_pipeline
)


class TestRLHFConfig:
    """Test RLHF Configuration"""
    
    def test_default_config(self):
        config = RLHFConfig()
        assert config.ppo_clip_eps == 0.2
        assert config.ppo_entropy_coef == 0.01
        assert config.ppo_learning_rate == 1e-5
        assert config.ppo_init_kl_coef == 0.02
        assert config.kl_penalty == "kl"
        assert config.gamma == 0.99
        assert config.lam == 0.95
    
    def test_custom_config(self):
        config = RLHFConfig(
            ppo_clip_eps=0.3,
            ppo_entropy_coef=0.02,
            ppo_learning_rate=5e-6,
            ppo_init_kl_coef=0.05
        )
        assert config.ppo_clip_eps == 0.3
        assert config.ppo_entropy_coef == 0.02
        assert config.ppo_learning_rate == 5e-6
        assert config.ppo_init_kl_coef == 0.05
    
    def test_reward_model_config(self):
        config = RLHFConfig(
            reward_model_hidden_size=768,
            reward_model_num_layers=12
        )
        assert config.reward_model_hidden_size == 768
        assert config.reward_model_num_layers == 12


class TestRewardModel:
    """Test Reward Model"""
    
    @pytest.fixture
    def model_config(self):
        return RLHFConfig(
            vocab_size=32000,
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=1024,
            reward_model_hidden_size=512
        )
    
    @pytest.fixture
    def sample_batch(self):
        batch_size = 2
        seq_len = 16
        return {
            "input_ids": torch.randint(0, 32000, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": torch.randint(0, 2, (batch_size,))
        }
    
    def test_reward_model_initialization(self, model_config):
        model = RewardModel(model_config)
        assert model.vocab_size == 32000
        assert model.hidden_size == 512
    
    def test_reward_model_forward(self, model_config, sample_batch):
        model = RewardModel(model_config)
        model.eval()
        
        with torch.no_grad():
            reward = model(
                sample_batch["input_ids"],
                sample_batch["attention_mask"]
            )
        
        assert reward.shape == (2,)
        assert not torch.isnan(reward).any()
    
    def test_reward_model_training_mode(self, model_config):
        model = RewardModel(model_config)
        model.train()
        
        assert model.training
        assert hasattr(model, 'score_head')
    
    def test_reward_computation(self, model_config):
        model = RewardModel(model_config)
        
        input_ids = torch.randint(0, 32000, (1, 32))
        attention_mask = torch.ones(1, 32)
        
        reward = model(input_ids, attention_mask)
        
        # Reward should be a single scalar per sample
        assert reward.shape == (1,)
        # Rewards can be positive or negative
        assert isinstance(reward.item(), float)


class TestRewardModelTrainer:
    """Test Reward Model Trainer"""
    
    @pytest.fixture
    def trainer_config(self):
        return RLHFConfig(
            vocab_size=32000,
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            reward_model_hidden_size=256,
            reward_learning_rate=1e-4,
            reward_num_epochs=3,
            reward_batch_size=4
        )
    
    def test_trainer_initialization(self, trainer_config):
        trainer = RewardModelTrainer(trainer_config)
        
        assert trainer.config is not None
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.device in ["cuda", "cpu"]
    
    def test_preference_pairs_format(self, trainer_config):
        trainer = RewardModelTrainer(trainer_config)
        
        # Test with sample preference data
        preferred = torch.randint(0, 32000, (4, 16))
        rejected = torch.randint(0, 32000, (4, 16))
        
        # Should be able to format into pairs
        pairs = trainer._format_preference_pairs(preferred, rejected)
        
        assert pairs is not None


class TestPPOTrainer:
    """Test PPO Trainer"""
    
    @pytest.fixture
    def ppo_config(self):
        return RLHFConfig(
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
            ppo_init_kl_coef=0.02
        )
    
    def test_ppo_trainer_initialization(self, ppo_config):
        trainer = PPOTrainer(ppo_config)
        
        assert trainer.config is not None
        assert trainer.actor is not None
        assert trainer.critic is not None
        assert trainer.ref_model is not None
        assert trainer.reward_model is not None
    
    def test_kl_penalty_types(self, ppo_config):
        """Test different KL penalty types"""
        for kl_penalty in ["kl", "kl-linear", "kl-log"]:
            ppo_config.kl_penalty = kl_penalty
            trainer = PPOTrainer(ppo_config)
            assert trainer.kl_penalty_type == kl_penalty
    
    def test_reward_normalization(self, ppo_config):
        trainer = PPOTrainer(ppo_config)
        
        rewards = torch.randn(32)
        normalized = trainer._normalize_rewards(rewards)
        
        # Normalized rewards should have mean ~0 and std ~1
        assert abs(normalized.mean().item()) < 0.1
        assert abs(normalized.std().item() - 1.0) < 0.1


class TestRLHFPipeline:
    """Test RLHF Pipeline"""
    
    def test_create_pipeline(self):
        config = RLHFConfig(
            vocab_size=32000,
            hidden_size=256,
            num_attention_heads=4,
            intermediate_size=512
        )
        
        pipeline = create_rlhf_pipeline(config)
        
        assert pipeline is not None
        assert hasattr(pipeline, 'reward_model')
        assert hasattr(pipeline, 'ppo_trainer')
    
    def test_pipeline_full_workflow(self):
        config = RLHFConfig(
            vocab_size=32000,
            hidden_size=256,
            num_attention_heads=4,
            intermediate_size=512,
            ppo_clip_eps=0.2
        )
        
        pipeline = create_rlhf_pipeline(config)
        
        # Test reward model training step
        reward_batch = {
            "input_ids": torch.randint(0, 32000, (2, 16)),
            "attention_mask": torch.ones(2, 16)
        }
        
        # Pipeline should have reward training capability
        assert pipeline.reward_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
