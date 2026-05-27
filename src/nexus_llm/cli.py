"""
Command Line Interface for Nexus-7B
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

import torch

from .model import NexusConfig, NexusForCausalLM
from .data import NexusTokenizer, DataProcessor, DataConfig
from .training import Trainer, DistributedTrainer, TrainingConfig, create_trainer
from .serving import InferenceEngine, ModelServer, InferenceConfig, create_model_server
from .enterprise import EnterpriseConfig, EnterpriseManager
from .utils import setup_logging, load_config, print_system_info, set_seed


logger = logging.getLogger(__name__)


def train_main():
    """Main entry point for training."""
    
    parser = argparse.ArgumentParser(description="Nexus-7B Training")
    
    # Config paths
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml",
                       help="Path to model configuration")
    parser.add_argument("--training-config", type=str, default="configs/training_config.yaml",
                       help="Path to training configuration")
    parser.add_argument("--data-config", type=str, default="configs/data_config.yaml",
                       help="Path to data configuration")
    parser.add_argument("--enterprise-config", type=str, default="configs/enterprise_config.yaml",
                       help="Path to enterprise configuration")
    
    # Override options
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory override")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--distributed", action="store_true",
                       help="Enable distributed training")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Print system info
    print_system_info()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configurations
    model_config_dict = load_config(args.model_config) if Path(args.model_config).exists() else {}
    training_config_dict = load_config(args.training_config) if Path(args.training_config).exists() else {}
    data_config_dict = load_config(args.data_config) if Path(args.data_config).exists() else {}
    enterprise_config_dict = load_config(args.enterprise_config) if Path(args.enterprise_config).exists() else {}
    
    # Create config objects
    model_config = NexusConfig(**model_config_dict.get('model', {}))
    training_config = TrainingConfig(**training_config_dict.get('training', {}))
    data_config = DataConfig(**data_config_dict.get('data', {}))
    enterprise_config = EnterpriseConfig(**enterprise_config_dict.get('enterprise', {}))
    
    # Override output dir
    if args.output_dir:
        training_config.output_dir = args.output_dir
    
    # Override resume
    if args.resume_from:
        training_config.resume_from_checkpoint = args.resume_from
    
    # Initialize enterprise features
    enterprise_manager = EnterpriseManager(enterprise_config)
    enterprise_manager.setup()
    
    # Create trainer
    trainer = create_trainer(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        distributed=args.distributed,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Shutdown
    enterprise_manager.shutdown()
    
    logger.info("Training completed successfully!")


def serve_main():
    """Main entry point for serving."""
    
    parser = argparse.ArgumentParser(description="Nexus-7B Serving")
    
    # Model paths
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                       help="Path to tokenizer")
    
    # Server options
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                       help="Server port")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of workers")
    
    # Inference options
    parser.add_argument("--precision", type=str, default="bf16",
                       choices=["fp32", "fp16", "bf16", "int8", "int4"],
                       help="Precision for inference")
    parser.add_argument("--quantize", action="store_true",
                       help="Enable quantization")
    
    # Enterprise options
    parser.add_argument("--enterprise-config", type=str, default=None,
                       help="Path to enterprise configuration")
    
    # Misc
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Print system info
    print_system_info()
    
    # Create inference config
    inference_config = InferenceConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        precision=args.precision,
        quantization_enabled=args.quantize,
        load_in_8bit=args.quantize and args.precision == "int8",
        load_in_4bit=args.quantize and args.precision == "int4",
    )
    
    # Load enterprise config
    enterprise_config = None
    if args.enterprise_config and Path(args.enterprise_config).exists():
        enterprise_config_dict = load_config(args.enterprise_config)
        enterprise_config = EnterpriseConfig(**enterprise_config_dict.get('enterprise', {}))
    
    # Create server
    server = create_model_server(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        enterprise_config=enterprise_config,
        inference_config=inference_config,
    )
    
    # Run server
    logger.info(f"Starting server on {args.host}:{args.port}")
    server.run(host=args.host, port=args.port, workers=args.workers)


def eval_main():
    """Main entry point for evaluation."""
    
    parser = argparse.ArgumentParser(description="Nexus-7B Evaluation")
    
    # Model paths
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                       help="Path to tokenizer")
    
    # Evaluation options
    parser.add_argument("--eval-file", type=str, required=True,
                       help="Path to evaluation data")
    parser.add_argument("--output-file", type=str, default="evaluation_results.json",
                       help="Path to output results")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Evaluation batch size")
    
    # Generation options
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Generation temperature (0 for greedy)")
    
    # Misc
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Create inference engine
    inference_config = InferenceConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    engine = InferenceEngine(
        model=NexusForCausalLM.from_pretrained(args.model_path),
        tokenizer=NexusTokenizer(DataConfig(tokenizer_path=args.tokenizer_path)),
        config=inference_config,
    )
    
    # Load evaluation data
    eval_data = []
    with open(args.eval_file, 'r') as f:
        for line in f:
            eval_data.append(json.loads(line))
    
    # Run evaluation
    results = []
    
    logger.info(f"Evaluating {len(eval_data)} samples...")
    
    for sample in eval_data:
        prompt = sample.get('prompt', sample.get('input', sample.get('text', '')))
        
        response = engine.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            return_full_response=True,
        )
        
        results.append({
            "prompt": prompt,
            "generated": response['text'],
            "reference": sample.get('reference', sample.get('output', sample.get('answer', ''))),
            "tokens_generated": response['tokens_generated'],
            "latency_ms": response['latency_ms'],
        })
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {args.output_file}")
    
    # Print summary
    avg_latency = sum(r['latency_ms'] for r in results) / len(results)
    avg_tokens = sum(r['tokens_generated'] for r in results) / len(results)
    
    print(f"\nEvaluation Summary:")
    print(f"  Total samples: {len(results)}")
    print(f"  Average latency: {avg_latency:.2f}ms")
    print(f"  Average tokens: {avg_tokens:.2f}")
    print(f"  Tokens/second: {avg_tokens / (avg_latency / 1000):.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: nexus-llm [train|serve|eval] [options]")
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = sys.argv[1:]
    
    if command == "train":
        train_main()
    elif command == "serve":
        serve_main()
    elif command == "eval":
        eval_main()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)