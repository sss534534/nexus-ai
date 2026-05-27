"""
Tests for Nexus-7B Model Architecture
"""

import pytest
import torch
import torch.nn as nn

from nexus_llm.model import NexusConfig, NexusModel, NexusForCausalLM, RMSNorm, RotaryPositionEmbedding


class TestNexusConfig:
    """Tests for NexusConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = NexusConfig()
        
        assert config.hidden_size == 4096
        assert config.num_attention_heads == 32
        assert config.num_hidden_layers == 32
        assert config.vocab_size == 152064
        assert config.max_position_embeddings == 8192
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = NexusConfig(
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=12,
        )
        
        assert config.hidden_size == 1024
        assert config.num_attention_heads == 16
        assert config.num_hidden_layers == 12
    
    def test_gqa_config(self):
        """Test Group Query Attention configuration."""
        config = NexusConfig(
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        
        assert config.num_key_value_heads == 8
        assert config.num_attention_groups == 4
    
    def test_invalid_gqa_config(self):
        """Test invalid GQA configuration."""
        with pytest.raises(ValueError):
            NexusConfig(
                num_attention_heads=32,
                num_key_value_heads=64,  # Should be <= num_attention_heads
            )


class TestRMSNorm:
    """Tests for RMSNorm layer."""
    
    def test_rms_norm_forward(self):
        """Test RMSNorm forward pass."""
        hidden_size = 256
        rms_norm = RMSNorm(hidden_size)
        
        x = torch.randn(2, 10, hidden_size)
        output = rms_norm(x)
        
        assert output.shape == x.shape
    
    def test_rms_norm_values(self):
        """Test RMSNorm normalization."""
        hidden_size = 256
        rms_norm = RMSNorm(hidden_size)
        
        x = torch.randn(1, 1, hidden_size)
        output = rms_norm(x)
        
        # Check that output is normalized
        variance = output.pow(2).mean(-1)
        expected_variance = torch.ones_like(variance) - rms_norm.eps
        
        assert torch.allclose(variance, expected_variance, atol=1e-5)


class TestRotaryPositionEmbedding:
    """Tests for RotaryPositionEmbedding."""
    
    def test_rope_forward(self):
        """Test RoPE forward pass."""
        config = NexusConfig(hidden_size=256, num_attention_heads=8)
        rope = RotaryPositionEmbedding(config)
        
        x = torch.randn(1, 8, 10, config.head_dim)
        output = rope(x)
        
        assert output.shape == x.shape
    
    def test_rope_rotation(self):
        """Test that RoPE applies rotation."""
        config = NexusConfig(hidden_size=256, num_attention_heads=8)
        rope = RotaryPositionEmbedding(config)
        
        x = torch.randn(1, 8, 10, config.head_dim)
        output = rope(x)
        
        # Output should be different from input
        assert not torch.allclose(output, x)
    
    def test_rope_dynamic_scaling(self):
        """Test dynamic scaling for extended positions."""
        config = NexusConfig(
            hidden_size=256,
            num_attention_heads=8,
            rope_scaling_type="dynamic",
            rope_scaling_factor=2.0,
        )
        rope = RotaryPositionEmbedding(config)
        
        x = torch.randn(1, 8, 100, config.head_dim)
        output = rope(x)
        
        assert output.shape == x.shape


class TestNexusModel:
    """Tests for NexusModel."""
    
    def test_model_creation(self):
        """Test model creation."""
        config = NexusConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=1000,
        )
        
        model = NexusModel(config)
        
        assert isinstance(model, nn.Module)
    
    def test_model_forward(self):
        """Test model forward pass."""
        config = NexusConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=1000,
        )
        
        model = NexusModel(config)
        
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        
        outputs = model(input_ids)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (1, 10, config.vocab_size)
    
    def test_model_with_attention_mask(self):
        """Test model with attention mask."""
        config = NexusConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=1000,
        )
        
        model = NexusModel(config)
        
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        attention_mask = torch.ones(1, 10)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        assert outputs["logits"].shape == (1, 10, config.vocab_size)
    
    def test_model_with_cache(self):
        """Test model with KV cache."""
        config = NexusConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=1000,
        )
        
        model = NexusModel(config)
        
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        
        outputs = model(input_ids, use_cache=True)
        
        assert "past_key_values" in outputs
        assert len(outputs["past_key_values"]) == config.num_hidden_layers
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing."""
        config = NexusConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=1000,
        )
        
        model = NexusModel(config)
        model.enable_gradient_checkpointing()
        
        assert model.gradient_checkpointing
        
        model.disable_gradient_checkpointing()
        
        assert not model.gradient_checkpointing
    
    def test_model_parameters_count(self):
        """Test model parameter count."""
        config = NexusConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=1000,
        )
        
        model = NexusModel(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Approximate calculation
        # Embedding: vocab_size * hidden_size
        embedding_params = config.vocab_size * config.hidden_size
        
        # Each layer: attention (4 * hidden^2) + MLP (3 * hidden * intermediate)
        layer_params = 4 * config.hidden_size * config.hidden_size + \
                       3 * config.hidden_size * config.intermediate_size
        
        total_layer_params = config.num_hidden_layers * layer_params
        
        expected_params = embedding_params + total_layer_params
        
        # Allow some tolerance for biases and norms
        assert abs(total_params - expected_params) < expected_params * 0.1


class TestNexusForCausalLM:
    """Tests for NexusForCausalLM."""
    
    def test_causal_lm_creation(self):
        """Test causal LM creation."""
        config = NexusConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=1000,
        )
        
        model = NexusForCausalLM(config)
        
        assert isinstance(model, nn.Module)
    
    def test_causal_lm_forward(self):
        """Test causal LM forward pass."""
        config = NexusConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=1000,
        )
        
        model = NexusForCausalLM(config)
        
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        
        outputs = model(input_ids)
        
        assert outputs[0].shape == (1, 10, config.vocab_size)
    
    def test_causal_lm_with_labels(self):
        """Test causal LM with labels."""
        config = NexusConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=1000,
        )
        
        model = NexusForCausalLM(config)
        
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        
        loss = outputs[0]
        logits = outputs[1]
        
        assert loss is not None
        assert loss.item() > 0  # Loss should be positive
        assert logits.shape == (1, 10, config.vocab_size)
    
    def test_causal_lm_generate(self):
        """Test causal LM generation."""
        config = NexusConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=1000,
        )
        
        model = NexusForCausalLM(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=1.0,
                do_sample=True,
            )
        
        assert output_ids.shape[1] >= input_ids.shape[1]
        assert output_ids.shape[1] <= input_ids.shape[1] + 10
    
    def test_causal_lm_greedy_generation(self):
        """Test greedy generation."""
        config = NexusConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=1000,
        )
        
        model = NexusForCausalLM(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,  # Greedy
            )
        
        assert output_ids.shape[1] >= input_ids.shape[1]


class TestModelDevicePlacement:
    """Tests for model device placement."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_placement(self):
        """Test CUDA device placement."""
        config = NexusConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=1000,
        )
        
        model = NexusModel(config)
        model = model.to("cuda")
        
        input_ids = torch.randint(0, config.vocab_size, (1, 10)).cuda()
        
        outputs = model(input_ids)
        
        assert outputs["logits"].device.type == "cuda"
    
    def test_cpu_placement(self):
        """Test CPU device placement."""
        config = NexusConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=1000,
        )
        
        model = NexusModel(config)
        model = model.to("cpu")
        
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        
        outputs = model(input_ids)
        
        assert outputs["logits"].device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])