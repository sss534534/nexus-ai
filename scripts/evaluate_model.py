#!/usr/bin/env python3
"""
模型评估脚本 - 困惑度、生成质量、性能基准测试
"""

import os
import json
import argparse
import logging
import time
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """评估配置"""
    model_path: str = "checkpoints/best"
    tokenizer_path: str = "tokenizer/nexus_tokenizer.model"
    eval_data_path: str = "data/prepared/test.jsonl"
    
    # 困惑度评估
    perplexity_batch_size: int = 8
    perplexity_max_samples: int = 1000
    
    # 生成评估
    generation_max_tokens: int = 256
    generation_temperature: float = 0.7
    generation_top_p: float = 0.95
    
    # 性能评估
    benchmark_iterations: int = 100
    benchmark_batch_sizes: List[int] = [1, 2, 4, 8]
    
    # 输出
    output_dir: str = "eval_results"


class PerplexityEvaluator:
    """困惑度评估器"""
    
    def __init__(self, model, tokenizer, config: EvalConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def evaluate(self, texts: List[str]) -> Dict[str, float]:
        """计算困惑度"""
        logger.info("Computing perplexity...")
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        batch_texts = []
        for i, text in enumerate(texts[:self.config.perplexity_max_samples]):
            batch_texts.append(text)
            
            if len(batch_texts) >= self.config.perplexity_batch_size:
                loss, tokens = self._compute_batch_loss(batch_texts)
                total_loss += loss
                total_tokens += tokens
                batch_texts = []
        
        # 处理剩余
        if batch_texts:
            loss, tokens = self._compute_batch_loss(batch_texts)
            total_loss += loss
            total_tokens += tokens
        
        # 计算困惑度
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        logger.info(f"Perplexity: {perplexity:.2f}")
        
        return {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
        }
    
    def _compute_batch_loss(self, texts: List[str]) -> tuple:
        """计算批次损失"""
        # 编码
        input_ids_list = []
        for text in texts:
            tokens = self.tokenizer.encode(text) if hasattr(self.tokenizer, 'encode') else text.split()
            input_ids_list.append(tokens)
        
        # 填充
        max_len = max(len(ids) for ids in input_ids_list)
        padded_ids = []
        attention_mask = []
        
        for ids in input_ids_list:
            padding = [0] * (max_len - len(ids))
            padded_ids.append(ids + padding)
            attention_mask.append([1] * len(ids) + padding)
        
        input_ids = torch.tensor(padded_ids)
        attention_mask = torch.tensor(attention_mask)
        
        # 计算损失
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs["logits"]
            
            # 计算交叉熵损失
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # 只计算有效token的损失
            valid_tokens = attention_mask[:, 1:].sum().item()
        
        return loss.item(), valid_tokens


class GenerationEvaluator:
    """生成质量评估器"""
    
    def __init__(self, model, tokenizer, config: EvalConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def evaluate(self, prompts: List[str], references: Optional[List[str]] = None) -> Dict[str, Any]:
        """评估生成质量"""
        logger.info("Evaluating generation quality...")
        
        self.model.eval()
        generations = []
        
        for prompt in prompts[:50]:  # 限制数量
            generated = self._generate(prompt)
            generations.append({
                "prompt": prompt,
                "generated": generated,
            })
        
        # 计算指标
        results = {
            "num_generations": len(generations),
            "avg_length": np.mean([len(g["generated"]) for g in generations]),
            "generations": generations[:10],  # 保存部分示例
        }
        
        # 如果有参考答案，计算相似度
        if references:
            similarities = self._compute_similarities(generations, references)
            results["avg_similarity"] = np.mean(similarities)
        
        logger.info(f"Average generation length: {results['avg_length']:.2f}")
        
        return results
    
    def _generate(self, prompt: str) -> str:
        """生成文本"""
        tokens = self.tokenizer.encode(prompt) if hasattr(self.tokenizer, 'encode') else prompt.split()
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            # 简化生成
            for _ in range(self.config.generation_max_tokens):
                outputs = self.model(input_ids)
                logits = outputs["logits"][:, -1, :]
                
                # 温度采样
                probs = torch.softmax(logits / self.config.generation_temperature, dim=-1)
                
                # Top-p 过滤
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.config.generation_top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[0, indices_to_remove] = 0
                
                # 采样
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # 检查结束
                if next_token.item() == 0:  # EOS
                    break
        
        # 解码
        generated_tokens = input_ids[0].tolist()
        generated_text = self.tokenizer.decode(generated_tokens) if hasattr(self.tokenizer, 'decode') else ' '.join(generated_tokens)
        
        return generated_text[len(prompt):]  # 去除prompt
    
    def _compute_similarities(self, generations: List[Dict], references: List[str]) -> List[float]:
        """计算相似度"""
        similarities = []
        
        for gen, ref in zip(generations, references):
            # 简化的相似度计算
            gen_words = set(gen["generated"].lower().split())
            ref_words = set(ref.lower().split())
            
            if gen_words and ref_words:
                intersection = len(gen_words & ref_words)
                union = len(gen_words | ref_words)
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        return similarities


class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self, model, tokenizer, config: EvalConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def run(self) -> Dict[str, Any]:
        """运行性能基准测试"""
        logger.info("Running performance benchmark...")
        
        self.model.eval()
        results = {}
        
        # 测试不同批次大小
        for batch_size in self.config.benchmark_batch_sizes:
            latency, throughput = self._benchmark_batch(batch_size)
            results[f"batch_{batch_size}"] = {
                "latency_ms": latency,
                "throughput_tokens_per_sec": throughput,
            }
        
        # 内存使用
        memory_stats = self._measure_memory()
        results["memory"] = memory_stats
        
        # 模型信息
        results["model_info"] = {
            "total_params": sum(p.numel() for p in self.model.parameters()),
            " trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        logger.info(f"Batch 1 latency: {results['batch_1']['latency_ms']:.2f} ms")
        logger.info(f"Batch 1 throughput: {results['batch_1']['throughput_tokens_per_sec']:.2f} tokens/sec")
        
        return results
    
    def _benchmark_batch(self, batch_size: int) -> tuple:
        """测试批次性能"""
        # 创建输入
        seq_len = 64
        vocab_size = 32000
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # 预热
        with torch.no_grad():
            for _ in range(5):
                self.model(input_ids)
        
        # 测量
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(self.config.benchmark_iterations):
                outputs = self.model(input_ids)
        
        elapsed = time.time() - start_time
        
        # 计算指标
        latency_ms = elapsed / self.config.benchmark_iterations * 1000
        throughput = batch_size * seq_len / elapsed
        
        return latency_ms, throughput
    
    def _measure_memory(self) -> Dict[str, float]:
        """测量内存使用"""
        memory_stats = {}
        
        if torch.cuda.is_available():
            memory_stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            memory_stats["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return memory_stats


class EvaluationRunner:
    """评估运行器"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
    
    def load_model(self):
        """加载模型"""
        logger.info(f"Loading model from {self.config.model_path}")
        
        # 这里使用简化模型创建
        from nexus_llm import NexusConfig, NexusForCausalLM
        
        config = NexusConfig(
            vocab_size=32000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
        )
        
        model = NexusForCausalLM(config)
        
        # 尝试加载权重
        model_path = Path(self.config.model_path)
        if model_path.exists():
            if model_path.suffix == '.pt' or model_path.suffix == '.bin':
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
                logger.info("Model weights loaded")
        
        return model
    
    def load_tokenizer(self):
        """加载分词器"""
        # 简化tokenizer
        class SimpleTokenizer:
            def encode(self, text):
                return [hash(word) % 32000 for word in text.split()]
            
            def decode(self, tokens):
                return ' '.join([f"tok_{t}" for t in tokens])
        
        return SimpleTokenizer()
    
    def load_eval_data(self) -> List[Dict]:
        """加载评估数据"""
        data_path = Path(self.config.eval_data_path)
        
        if not data_path.exists():
            logger.warning(f"Eval data not found: {data_path}")
            return []
        
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(data)} evaluation samples")
        return data
    
    def run_all(self) -> Dict[str, Any]:
        """运行所有评估"""
        # 加载模型和数据
        model = self.load_model()
        tokenizer = self.load_tokenizer()
        eval_data = self.load_eval_data()
        
        results = {}
        
        # 困惑度评估
        if eval_data:
            texts = [item.get("text", "") for item in eval_data]
            perplexity_eval = PerplexityEvaluator(model, tokenizer, self.config)
            results["perplexity"] = perplexity_eval.evaluate(texts)
        
        # 生成评估
        prompts = [item.get("prompt", item.get("text", "")) for item in eval_data[:20]]
        if prompts:
            generation_eval = GenerationEvaluator(model, tokenizer, self.config)
            results["generation"] = generation_eval.evaluate(prompts)
        
        # 性能基准
        benchmark = PerformanceBenchmark(model, tokenizer, self.config)
        results["benchmark"] = benchmark.run()
        
        # 保存结果
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict):
        """保存评估结果"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "eval_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument("--model-path", default="checkpoints/best", help="模型路径")
    parser.add_argument("--tokenizer-path", default="tokenizer/nexus_tokenizer.model", help="分词器路径")
    parser.add_argument("--eval-data", default="data/prepared/test.jsonl", help="评估数据路径")
    parser.add_argument("--output-dir", default="eval_results", help="输出目录")
    parser.add_argument("--perplexity-samples", type=int, default=1000, help="困惑度评估样本数")
    parser.add_argument("--benchmark-iterations", type=int, default=100, help="基准测试迭代次数")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        perplexity_max_samples=args.perplexity_samples,
        benchmark_iterations=args.benchmark_iterations,
    )
    
    runner = EvaluationRunner(config)
    results = runner.run_all()
    
    # 打印摘要
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    
    if "perplexity" in results:
        print(f"Perplexity: {results['perplexity']['perplexity']:.2f}")
    
    if "benchmark" in results:
        print(f"Batch 1 Latency: {results['benchmark']['batch_1']['latency_ms']:.2f} ms")
        print(f"Total Parameters: {results['benchmark']['model_info']['total_params']:,}")
    
    print("=" * 50)


if __name__ == "__main__":
    main()