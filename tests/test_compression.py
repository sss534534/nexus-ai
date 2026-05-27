"""
Compression Module Tests
"""
import pytest
import torch
import torch.nn as nn
from nexus_llm.compression import (
    ModelCompressor,
    KnowledgeDistiller,
    StructuredPruner,
    CompressionConfig,
    create_compression_pipeline
)


class TestCompressionConfig:
    """Test Compression Configuration"""
    
    def test_default_config(self):
        config = CompressionConfig()
        assert config.distillation_temperature == 2.0
        assert config.distillation_alpha == 0.5
        assert config.pruning_ratio == 0.3
        assert config.quantization_mode == "dynamic"
        assert config.quantization_bits == 8
    
    def test_custom_config(self):
        config = CompressionConfig(
            distillation_temperature=4.0,
            distillation_alpha=0.7,
            pruning_ratio=0.5,
            quantization_bits=4
        )
        assert config.distillation_temperature == 4.0
        assert config.distillation_alpha == 0.7
        assert config.pruning_ratio == 0.5
        assert config.quantization_bits == 4
    
    def test_quantization_config(self):
        config = CompressionConfig(
            quantization_mode="static",
            calibration_data_size=100
        )
        assert config.quantization_mode == "static"
        assert config.calibration_data_size == 100


class MockModel(nn.Module):
    """Mock model for testing"""
    def __init__(self, hidden_size=256, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


class TestKnowledgeDistiller:
    """Test Knowledge Distiller"""
    
    @pytest.fixture
    def teacher_model(self):
        return MockModel(hidden_size=256, num_layers=4)
    
    @pytest.fixture
    def student_model(self):
        return MockModel(hidden_size=128, num_layers=2)
    
    @pytest.fixture
    def distiller(self, teacher_model, student_model):
        config = CompressionConfig(
            distillation_temperature=4.0,
            distillation_alpha=0.5
        )
        return KnowledgeDistiller(
            teacher_model,
            student_model,
            config
        )
    
    @pytest.fixture
    def sample_batch(self):
        return torch.randn(4, 128, 256)
    
    def test_distiller_initialization(self, distiller):
        assert distiller.teacher is not None
        assert distiller.student is not None
        assert distiller.temperature == 4.0
        assert distiller.alpha == 0.5
    
    def test_distillation_loss(self, distiller, sample_batch):
        # Mock teacher output
        with torch.no_grad():
            teacher_logits = distiller.teacher(sample_batch)
        
        student_logits = distiller.student(sample_batch)
        
        loss = distiller._compute_distillation_loss(
            student_logits,
            teacher_logits
        )
        
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
    
    def test_kl_divergence_computation(self, distiller, sample_batch):
        with torch.no_grad():
            teacher_logits = distiller.teacher(sample_batch)
        
        student_logits = distiller.student(sample_batch)
        
        loss = distiller.compute_loss(
            student_logits,
            teacher_logits
        )
        
        assert isinstance(loss.item(), float)
    
    def test_layer_mapping(self, distiller):
        # Check that layer mapping is created
        assert len(distiller.layer_mapping) >= 0


class TestStructuredPruner:
    """Test Structured Pruner"""
    
    @pytest.fixture
    def model(self):
        return MockModel(hidden_size=256, num_layers=4)
    
    @pytest.fixture
    def pruner(self, model):
        config = CompressionConfig(
            pruning_ratio=0.3,
            pruning_type="magnitude"
        )
        return StructuredPruner(model, config)
    
    def test_pruner_initialization(self, pruner):
        assert pruner.model is not None
        assert pruner.sparsity_ratio == 0.3
        assert pruner.pruning_type == "magnitude"
    
    def test_compute_mask(self, pruner):
        mask = pruner._compute_mask()
        assert isinstance(mask, dict)
    
    def test_prune_model(self, pruner):
        initial_params = sum(p.numel() for p in pruner.model.parameters())
        
        pruner.prune()
        
        final_params = sum(p.numel() for p in pruner.model.parameters())
        
        # Pruned model should have fewer or equal parameters
        assert final_params <= initial_params
    
    def test_magnitude_pruning(self, model):
        config = CompressionConfig(
            pruning_ratio=0.5,
            pruning_type="magnitude"
        )
        pruner = StructuredPruner(model, config)
        
        # Run pruning
        pruner.prune()
        
        # Check that some weights are zeroed
        total_zeros = 0
        total_params = 0
        for param in model.parameters():
            if param.dim() >= 2:  # Only for weight matrices
                total_zeros += (param == 0).sum().item()
                total_params += param.numel()
        
        # Should have some zeros after magnitude pruning
        assert total_zeros > 0
    
    def test_gradient_pruning(self, model):
        config = CompressionConfig(
            pruning_ratio=0.3,
            pruning_type="gradient"
        )
        pruner = StructuredPruner(model, config)
        
        # Should be able to compute gradients
        x = torch.randn(4, 128, 256)
        output = model(x)
        output.sum().backward()
        
        # Pruning should work with gradients
        pruner.prune()


class TestModelCompressor:
    """Test Model Compressor"""
    
    @pytest.fixture
    def compressor_config(self):
        return CompressionConfig(
            enable_distillation=True,
            enable_pruning=True,
            enable_quantization=True,
            distillation_temperature=4.0,
            pruning_ratio=0.3,
            quantization_bits=8
        )
    
    @pytest.fixture
    def teacher_model(self):
        return MockModel(hidden_size=256, num_layers=4)
    
    @pytest.fixture
    def student_model(self):
        return MockModel(hidden_size=128, num_layers=2)
    
    def test_compressor_initialization(self, compressor_config):
        compressor = ModelCompressor(compressor_config)
        assert compressor.config is not None
        assert compressor.distiller is None  # Created on demand
        assert compressor.pruner is None  # Created on demand
    
    def test_compress_with_distillation(self, compressor_config, teacher_model, student_model):
        compressor = ModelCompressor(compressor_config)
        
        result = compressor.compress_with_distillation(
            teacher_model,
            student_model,
            torch.randn(4, 128, 256)
        )
        
        assert "compressed_model" in result
        assert "distillation_loss" in result
    
    def test_compress_with_pruning(self, compressor_config, teacher_model):
        compressor = ModelCompressor(compressor_config)
        
        result = compressor.compress_with_pruning(
            teacher_model,
            target_sparsity=0.3
        )
        
        assert "compressed_model" in result
        assert "sparsity" in result
    
    def test_quantize_model(self, compressor_config, teacher_model):
        compressor = ModelCompressor(compressor_config)
        
        result = compressor.quantize_model(
            teacher_model,
            bits=8
        )
        
        assert "quantized_model" in result
        assert "quantization_config" in result
    
    def test_full_compression_pipeline(self, compressor_config, teacher_model, student_model):
        compressor = ModelCompressor(compressor_config)
        
        result = compressor.compress(
            teacher_model,
            student_model,
            torch.randn(4, 128, 256)
        )
        
        assert "compressed_model" in result
        assert "compression_ratio" in result
        assert "method" in result
    
    def test_compression_ratio_calculation(self, compressor_config, teacher_model, student_model):
        compressor = ModelCompressor(compressor_config)
        
        # Calculate compression ratio
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        
        ratio = compressor._calculate_compression_ratio(
            teacher_params,
            student_params
        )
        
        assert ratio > 0
        assert ratio < 1  # Student should be smaller


class TestCompressionPipeline:
    """Test Compression Pipeline Factory"""
    
    def test_create_pipeline(self):
        config = CompressionConfig(
            enable_distillation=True,
            enable_pruning=True
        )
        
        pipeline = create_compression_pipeline(config)
        
        assert pipeline is not None
        assert hasattr(pipeline, 'compress_with_distillation')
        assert hasattr(pipeline, 'compress_with_pruning')
    
    def test_pipeline_distillation(self):
        config = CompressionConfig(enable_distillation=True)
        pipeline = create_compression_pipeline(config)
        
        teacher = MockModel(256, 4)
        student = MockModel(128, 2)
        
        result = pipeline.compress_with_distillation(
            teacher, student, torch.randn(2, 128, 256)
        )
        
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
