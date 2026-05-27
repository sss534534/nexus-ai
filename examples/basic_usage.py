"""
Example: Basic Model Usage
"""

import torch
from nexus_llm import NexusModel, NexusConfig, NexusTokenizer
from nexus_llm.data import DataConfig
from nexus_llm.serving import InferenceEngine, InferenceConfig


def main():
    # Create model configuration
    config = NexusConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=32,  # Set to 8 for GQA
        vocab_size=152064,
        max_position_embeddings=8192,
        use_flash_attention=True,
    )
    
    print("Model Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Vocabulary size: {config.vocab_size}")
    
    # Create model
    model = NexusModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 2 / (1024**3):.2f} GB (BF16)")
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"\nModel loaded on: {device}")
    
    # Test forward pass
    input_ids = torch.randint(0, config.vocab_size, (1, 100)).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    
    # Generate sample
    print("\nGenerating sample text...")
    
    # Note: For actual generation, you need a trained model and tokenizer
    # This is a demonstration of the API
    
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    
    print(f"  Generated sequence length: {generated_ids.shape[1]}")


def inference_example():
    """Example of using the inference engine."""
    
    # Configuration
    inference_config = InferenceConfig(
        model_path="checkpoints/best",
        tokenizer_path="tokenizer/nexus_tokenizer.model",
        precision="bf16",
        max_new_tokens=512,
        temperature=0.7,
    )
    
    # Note: This requires a trained model
    # Uncomment when you have a trained model
    
    # from nexus_llm.serving import create_inference_engine
    # 
    # engine = create_inference_engine(
    #     model_path="checkpoints/best",
    #     tokenizer_path="tokenizer/nexus_tokenizer.model",
    #     config=inference_config,
    # )
    # 
    # # Generate text
    # response = engine.generate(
    #     prompt="什么是深度学习？请详细解释。",
    #     max_new_tokens=256,
    #     temperature=0.7,
    #     return_full_response=True,
    # )
    # 
    # print("Generated text:")
    # print(response["text"])
    # print(f"\nTokens generated: {response['tokens_generated']}")
    # print(f"Latency: {response['latency_ms']:.2f} ms")
    # print(f"Speed: {response['tokens_per_second']:.2f} tokens/s")
    
    print("Inference example - requires trained model")


if __name__ == "__main__":
    print("=" * 50)
    print("Nexus-7B Basic Usage Example")
    print("=" * 50)
    
    main()
    
    print("\n" + "=" * 50)
    print("Inference Engine Example")
    print("=" * 50)
    
    inference_example()