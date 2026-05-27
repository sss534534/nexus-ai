"""
Evaluation and Benchmarking Module for Nexus-7B
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import statistics

import torch
import numpy as np
from tqdm import tqdm

from ..model import NexusForCausalLM
from ..data import NexusTokenizer
from ..serving import InferenceEngine, InferenceConfig


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    
    # Basic metrics
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Generation metrics
    bleu: Optional[float] = None
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None
    
    # Performance metrics
    latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    num_samples: int = 0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary."""
        return cls(**data)


class PerplexityEvaluator:
    """Evaluate model perplexity."""
    
    def __init__(self, model: NexusForCausalLM, tokenizer: NexusTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def evaluate(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 2048,
    ) -> EvaluationResult:
        """Evaluate perplexity on text corpus."""
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating perplexity"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encodings = self.tokenizer.batch_encode(
                    batch_texts,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                )
                
                input_ids = encodings["input_ids"].to(self.model.device)
                attention_mask = encodings["attention_mask"].to(self.model.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                
                loss = outputs[0] if isinstance(outputs, tuple) else outputs["loss"]
                
                # Accumulate
                num_tokens = attention_mask.sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return EvaluationResult(
            perplexity=perplexity,
            num_samples=len(texts),
        )


class GenerationEvaluator:
    """Evaluate text generation quality."""
    
    def __init__(self, engine: InferenceEngine):
        self.engine = engine
    
    def evaluate(
        self,
        test_data: List[Dict[str, str]],
        metrics: List[str] = None,
    ) -> EvaluationResult:
        """Evaluate generation quality."""
        
        metrics = metrics or ["bleu", "rouge"]
        
        predictions = []
        references = []
        latencies = []
        
        for item in tqdm(test_data, desc="Generating"):
            prompt = item.get("prompt", item.get("input", ""))
            reference = item.get("reference", item.get("output", item.get("answer", "")))
            
            # Generate
            start_time = time.time()
            response = self.engine.generate(
                prompt,
                max_new_tokens=256,
                temperature=0.0,  # Greedy for evaluation
            )
            latency = (time.time() - start_time) * 1000
            
            predictions.append(response["text"])
            references.append(reference)
            latencies.append(latency)
        
        result = EvaluationResult(
            num_samples=len(test_data),
            latency_ms=statistics.mean(latencies),
        )
        
        # Calculate metrics
        if "bleu" in metrics:
            result.bleu = self._calculate_bleu(predictions, references)
        
        if "rouge" in metrics:
            rouge_scores = self._calculate_rouge(predictions, references)
            result.rouge1 = rouge_scores["rouge1"]
            result.rouge2 = rouge_scores["rouge2"]
            result.rougeL = rouge_scores["rougeL"]
        
        return result
    
    def _calculate_bleu(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Calculate BLEU score."""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            scores = []
            smoothing = SmoothingFunction()
            
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                
                score = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    smoothing_function=smoothing.method1,
                )
                scores.append(score)
            
            return statistics.mean(scores)
        
        except ImportError:
            logger.warning("nltk not available for BLEU calculation")
            return 0.0
    
    def _calculate_rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            scores = defaultdict(list)
            
            for pred, ref in zip(predictions, references):
                score = scorer.score(ref, pred)
                for key in ['rouge1', 'rouge2', 'rougeL']:
                    scores[key].append(score[key].fmeasure)
            
            return {
                "rouge1": statistics.mean(scores["rouge1"]),
                "rouge2": statistics.mean(scores["rouge2"]),
                "rougeL": statistics.mean(scores["rougeL"]),
            }
        
        except ImportError:
            logger.warning("rouge-score not available for ROUGE calculation")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


class BenchmarkSuite:
    """Comprehensive benchmark suite."""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self.engine = None
    
    def load_model(self, device: str = "cuda"):
        """Load model for evaluation."""
        from ..model import NexusConfig
        from ..data import DataConfig
        
        # Load config
        config_file = Path(self.model_path) / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config_dict = json.load(f)
            config = NexusConfig(**config_dict)
        else:
            config = NexusConfig()
        
        # Load model
        self.model = NexusForCausalLM(config)
        self.model.load_state_dict(
            torch.load(Path(self.model_path) / "pytorch_model.bin", map_location="cpu")
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        # Load tokenizer
        data_config = DataConfig(tokenizer_path=self.tokenizer_path)
        self.tokenizer = NexusTokenizer(data_config)
        
        # Create inference engine
        inference_config = InferenceConfig(
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
        )
        self.engine = InferenceEngine(self.model, self.tokenizer, inference_config)
        
        logger.info("Model loaded for evaluation")
    
    def run_benchmark(
        self,
        benchmark_name: str,
        data_path: Optional[str] = None,
    ) -> EvaluationResult:
        """Run a specific benchmark."""
        
        if benchmark_name == "perplexity":
            return self._run_perplexity_benchmark(data_path)
        elif benchmark_name == "generation":
            return self._run_generation_benchmark(data_path)
        elif benchmark_name == "performance":
            return self._run_performance_benchmark()
        elif benchmark_name == "all":
            return self._run_all_benchmarks(data_path)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    def _run_perplexity_benchmark(self, data_path: Optional[str]) -> EvaluationResult:
        """Run perplexity benchmark."""
        
        if data_path is None:
            # Use default test data
            data_path = "data/test.jsonl"
        
        # Load test data
        texts = []
        with open(data_path) as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", "")
                texts.append(text)
        
        # Evaluate
        evaluator = PerplexityEvaluator(self.model, self.tokenizer)
        result = evaluator.evaluate(texts)
        
        logger.info(f"Perplexity: {result.perplexity:.2f}")
        
        return result
    
    def _run_generation_benchmark(self, data_path: Optional[str]) -> EvaluationResult:
        """Run generation quality benchmark."""
        
        if data_path is None:
            data_path = "data/test_generation.jsonl"
        
        # Load test data
        test_data = []
        with open(data_path) as f:
            for line in f:
                test_data.append(json.loads(line))
        
        # Evaluate
        evaluator = GenerationEvaluator(self.engine)
        result = evaluator.evaluate(test_data)
        
        logger.info(f"BLEU: {result.bleu:.4f}")
        logger.info(f"ROUGE-L: {result.rougeL:.4f}")
        
        return result
    
    def _run_performance_benchmark(self) -> EvaluationResult:
        """Run performance benchmark."""
        
        test_prompts = [
            "What is artificial intelligence?",
            "Explain machine learning.",
            "How does deep learning work?",
        ]
        
        latencies = []
        tokens_generated = []
        
        for prompt in test_prompts:
            start_time = time.time()
            response = self.engine.generate(
                prompt,
                max_new_tokens=128,
            )
            latency = (time.time() - start_time) * 1000
            
            latencies.append(latency)
            tokens_generated.append(response["tokens_generated"])
        
        avg_latency = statistics.mean(latencies)
        avg_tokens = statistics.mean(tokens_generated)
        tokens_per_second = avg_tokens / (avg_latency / 1000)
        
        # Memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            memory_usage = 0.0
        
        result = EvaluationResult(
            latency_ms=avg_latency,
            tokens_per_second=tokens_per_second,
            memory_usage_mb=memory_usage,
            num_samples=len(test_prompts),
        )
        
        logger.info(f"Latency: {avg_latency:.2f} ms")
        logger.info(f"Tokens/s: {tokens_per_second:.2f}")
        logger.info(f"Memory: {memory_usage:.2f} MB")
        
        return result
    
    def _run_all_benchmarks(self, data_path: Optional[str]) -> Dict[str, EvaluationResult]:
        """Run all benchmarks."""
        
        results = {}
        
        results["perplexity"] = self._run_perplexity_benchmark(data_path)
        results["generation"] = self._run_generation_benchmark(data_path)
        results["performance"] = self._run_performance_benchmark()
        
        return results
    
    def save_results(
        self,
        results: Dict[str, EvaluationResult],
        output_path: str,
    ):
        """Save benchmark results."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        results_dict = {
            name: result.to_dict()
            for name, result in results.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


class BenchmarkRegistry:
    """Registry for benchmark datasets."""
    
    BENCHMARKS = {
        "ceval": {
            "name": "C-Eval",
            "description": "Chinese Evaluation Suite",
            "url": "https://cevalbenchmark.com/",
            "language": "zh",
        },
        "cmmlu": {
            "name": "CMMLU",
            "description": "Chinese Massive Multitask Language Understanding",
            "url": "https://github.com/haonan-li/CMMLU",
            "language": "zh",
        },
        "mmlu": {
            "name": "MMLU",
            "description": "Massive Multitask Language Understanding",
            "url": "https://github.com/hendrycks/test",
            "language": "en",
        },
        "gsm8k": {
            "name": "GSM8K",
            "description": "Grade School Math",
            "url": "https://github.com/openai/grade-school-math",
            "language": "en",
        },
        "humaneval": {
            "name": "HumanEval",
            "description": "Code Generation Benchmark",
            "url": "https://github.com/openai/human-eval",
            "language": "code",
        },
        "bbh": {
            "name": "BBH",
            "description": "Big Bench Hard",
            "url": "https://github.com/suzgunmirac/BIG-Bench-Hard",
            "language": "en",
        },
    }
    
    @classmethod
    def list_benchmarks(cls) -> List[Dict[str, str]]:
        """List available benchmarks."""
        return [
            {"id": k, **v}
            for k, v in cls.BENCHMARKS.items()
        ]
    
    @classmethod
    def get_benchmark(cls, benchmark_id: str) -> Optional[Dict[str, str]]:
        """Get benchmark information."""
        return cls.BENCHMARKS.get(benchmark_id)
    
    @classmethod
    def download_benchmark(cls, benchmark_id: str, output_dir: str):
        """Download benchmark dataset."""
        
        benchmark = cls.BENCHMARKS.get(benchmark_id)
        if not benchmark:
            raise ValueError(f"Unknown benchmark: {benchmark_id}")
        
        logger.info(f"Downloading {benchmark['name']} from {benchmark['url']}")
        
        # This would implement actual download logic
        # For now, just create directory
        output_path = Path(output_dir) / benchmark_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Benchmark downloaded to {output_path}")


# Convenience functions
def evaluate_perplexity(
    model: NexusForCausalLM,
    tokenizer: NexusTokenizer,
    texts: List[str],
) -> float:
    """Evaluate perplexity on texts."""
    evaluator = PerplexityEvaluator(model, tokenizer)
    result = evaluator.evaluate(texts)
    return result.perplexity


def run_benchmark(
    model_path: str,
    tokenizer_path: str,
    benchmark: str = "all",
    output_path: Optional[str] = None,
) -> Dict[str, EvaluationResult]:
    """Run benchmark suite."""
    
    suite = BenchmarkSuite(model_path, tokenizer_path)
    suite.load_model()
    
    if benchmark == "all":
        results = suite._run_all_benchmarks()
    else:
        results = {benchmark: suite.run_benchmark(benchmark)}
    
    if output_path:
        suite.save_results(results, output_path)
    
    return results
