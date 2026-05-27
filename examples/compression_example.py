"""
Model Compression Example
Demonstrates how to use the compression module
"""
import torch
import torch.nn as nn
from nexus_llm.compression import (
    CompressionConfig,
    ModelCompressor,
    KnowledgeDistiller,
    StructuredPruner,
    create_compression_pipeline
)


# Model Definitions for Testing
class TeacherModel(nn.Module):
    """Larger teacher model"""
    def __init__(self, vocab_size=32000, hidden_size=512, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class StudentModel(nn.Module):
    """Smaller student model"""
    def __init__(self, vocab_size=32000, hidden_size=256, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def example_knowledge_distillation():
    """Example: Knowledge Distillation"""
    print("=" * 50)
    print("Example: Knowledge Distillation")
    print("=" * 50)
    
    # Create models
    teacher = TeacherModel(vocab_size=1000, hidden_size=256, num_layers=4)
    student = StudentModel(vocab_size=1000, hidden_size=128, num_layers=2)
    
    # Configure distillation
    config = CompressionConfig(
        distillation_temperature=4.0,
        distillation_alpha=0.7,  # Weight for KL loss
        enable_distillation=True
    )
    
    # Create distiller
    distiller = KnowledgeDistiller(teacher, student, config)
    
    print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")
    print(f"Temperature: {distiller.temperature}")
    print(f"Alpha: {distiller.alpha}")
    
    # Create sample batch
    batch_size = 4
    seq_len = 32
    batch = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Get teacher logits (no grad)
    with torch.no_grad():
        teacher_logits = teacher(batch)
    
    # Get student logits
    student_logits = student(batch)
    
    # Compute distillation loss
    loss = distiller.compute_loss(student_logits, teacher_logits)
    
    print(f"\nDistillation loss: {loss.item():.4f}")
    
    # Full compression
    result = distiller.compress(batch)
    
    print(f"\nCompressed model ready!")
    print(f"Final loss: {result.get('loss', loss.item()):.4f}")
    
    return distiller


def example_structured_pruning():
    """Example: Structured Pruning"""
    print("\n" + "=" * 50)
    print("Example: Structured Pruning")
    print("=" * 50)
    
    # Create model to prune
    model = TeacherModel(vocab_size=1000, hidden_size=128, num_layers=2)
    
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {initial_params:,}")
    
    # Configure pruning
    config = CompressionConfig(
        pruning_ratio=0.3,  # Remove 30% of weights
        pruning_type="magnitude",
        enable_pruning=True
    )
    
    # Create pruner
    pruner = StructuredPruner(model, config)
    
    print(f"Pruning type: {pruner.pruning_type}")
    print(f"Target sparsity: {pruner.sparsity_ratio}")
    
    # Perform pruning
    pruner.prune()
    
    # Count zeroed parameters
    zero_count = sum((p == 0).sum().item() for p in model.parameters())
    final_params = sum(p.numel() for p in model.parameters())
    
    actual_sparsity = zero_count / final_params
    
    print(f"\nAfter pruning:")
    print(f"  Zero parameters: {zero_count:,}")
    print(f"  Total parameters: {final_params:,}")
    print(f"  Actual sparsity: {actual_sparsity:.2%}")
    print(f"  Compression ratio: {final_params / initial_params:.2%}")
    
    return pruner


def example_quantization():
    """Example: Model Quantization"""
    print("\n" + "=" * 50)
    print("Example: Model Quantization")
    print("=" * 50)
    
    # Create model to quantize
    model = TeacherModel(vocab_size=1000, hidden_size=64, num_layers=1)
    
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Original model size: {original_size / 1024:.2f} KB")
    
    # Configure quantization
    config = CompressionConfig(
        quantization_mode="dynamic",  # Dynamic quantization
        quantization_bits=8,
        enable_quantization=True
    )
    
    # Create compressor
    compressor = ModelCompressor(config)
    
    # Quantize model
    result = compressor.quantize_model(model, bits=8)
    
    print(f"Quantization mode: {config.quantization_mode}")
    print(f"Bits: {config.quantization_bits}")
    
    quantized_model = result["quantized_model"]
    
    # Estimate quantized size
    quantized_size = sum(p.numel() * 1 for p in quantized_model.parameters())  # INT8 = 1 byte
    
    print(f"Quantized model size: {quantized_size / 1024:.2f} KB")
    print(f"Size reduction: {original_size / quantized_size:.2f}x")
    
    return compressor


def example_full_compression():
    """Example: Full Compression Pipeline"""
    print("\n" + "=" * 50)
    print("Example: Full Compression Pipeline")
    print("=" * 50)
    
    # Create models
    teacher = TeacherModel(vocab_size=1000, hidden_size=256, num_layers=4)
    student = StudentModel(vocab_size=1000, hidden_size=128, num_layers=2)
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    
    # Full compression config
    config = CompressionConfig(
        enable_distillation=True,
        enable_pruning=True,
        enable_quantization=True,
        distillation_temperature=4.0,
        distillation_alpha=0.5,
        pruning_ratio=0.2,
        quantization_bits=8
    )
    
    # Create compressor
    compressor = ModelCompressor(config)
    
    # Sample batch
    batch = torch.randint(0, 1000, (4, 32))
    
    # Run full compression
    result = compressor.compress(teacher, student, batch)
    
    print("\nCompression Results:")
    print(f"  Method: {result.get('method', 'N/A')}")
    print(f"  Compression ratio: {result.get('compression_ratio', 0):.2%}")
    print(f"  Distillation loss: {result.get('distillation_loss', 'N/A')}")
    
    compressed_model = result.get("compressed_model")
    if compressed_model:
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        print(f"  Compressed params: {compressed_params:,}")
    
    return compressor


def example_compression_pipeline():
    """Example: Compression Pipeline Factory"""
    print("\n" + "=" * 50)
    print("Example: Compression Pipeline")
    print("=" * 50)
    
    # Create pipeline
    config = CompressionConfig(
        enable_distillation=True,
        enable_pruning=True
    )
    
    pipeline = create_compression_pipeline(config)
    
    print("Pipeline components:")
    print(f"  Distillation: {config.enable_distillation}")
    print(f"  Pruning: {config.enable_pruning}")
    print(f"  Quantization: {config.enable_quantization}")
    
    # Use pipeline
    teacher = TeacherModel(vocab_size=1000, hidden_size=128, num_layers=2)
    student = StudentModel(vocab_size=1000, hidden_size=64, num_layers=1)
    
    batch = torch.randint(0, 1000, (2, 16))
    
    result = pipeline.compress_with_distillation(teacher, student, batch)
    
    print(f"\nPipeline result:")
    print(f"  Status: {'Success' if result.get('compressed_model') is not None else 'Failed'}")
    print(f"  Loss: {result.get('distillation_loss', 'N/A'):.4f}")
    
    return pipeline


def example_pruning_methods():
    """Example: Different Pruning Methods"""
    print("\n" + "=" * 50)
    print("Example: Pruning Methods Comparison")
    print("=" * 50)
    
    methods = ["magnitude", "gradient"]
    
    for method in methods:
        # Create fresh model
        model = TeacherModel(vocab_size=500, hidden_size=64, num_layers=1)
        
        config = CompressionConfig(
            pruning_ratio=0.3,
            pruning_type=method,
            enable_pruning=True
        )
        
        pruner = StructuredPruner(model, config)
        
        # Compute gradients for gradient-based pruning
        if method == "gradient":
            x = torch.randint(0, 500, (2, 16))
            output = model(x)
            output.sum().backward()
        
        # Prune
        pruner.prune()
        
        # Count sparsity
        zero_count = sum((p == 0).sum().item() for p in model.parameters())
        total = sum(p.numel() for p in model.parameters())
        sparsity = zero_count / total
        
        print(f"  {method.capitalize():12s}: Sparsity = {sparsity:.2%}")
    
    print("\nNote: Magnitude pruning removes small weights.")
    print("      Gradient pruning removes weights with small gradients.")


if __name__ == "__main__":
    # Run all examples
    distiller = example_knowledge_distillation()
    pruner = example_structured_pruning()
    compressor = example_quantization()
    full_compression = example_full_compression()
    pipeline = example_compression_pipeline()
    example_pruning_methods()
    
    print("\n" + "=" * 50)
    print("All Compression examples completed!")
    print("=" * 50)
