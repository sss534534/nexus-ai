#!/usr/bin/env python3
"""
数据准备脚本 - 下载和处理公开数据集用于小模型训练
支持多种数据源：HuggingFace、本地文件、自定义数据
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
import random

try:
    from datasets import load_dataset, Dataset
    from huggingface_hub import hf_hub_download, list_datasets
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DataPrepConfig:
    """数据准备配置"""
    # 输出配置
    output_dir: str = "data/prepared"
    output_format: str = "jsonl"  # jsonl, parquet, hf
    
    # 数据集配置
    datasets: List[str] = None  # HuggingFace 数据集列表
    max_samples: int = 100000  # 每个数据集最大样本数
    min_length: int = 50  # 最小文本长度
    max_length: int = 2048  # 最大文本长度
    
    # 预处理配置
    deduplicate: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    normalize_whitespace: bool = True
    filter_language: str = "zh"  # zh, en, all
    
    # 分词配置
    tokenizer_path: Optional[str] = None
    vocab_size: int = 32000
    
    # 划分配置
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = [
                # 中文数据集
                "shibing624/alpaca-zh",
                "FreedomIntelligence/HuatuoGPT-sft-data-v1",
                # 英文数据集
                "tatsu-lab/alpaca",
                "OpenAssistant/oasst1",
            ]


class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
    
    def preprocess(self, text: str) -> str:
        """预处理单条文本"""
        if not text or len(text) < self.config.min_length:
            return ""
        
        # 去除URL
        if self.config.remove_urls:
            import re
            text = re.sub(r'https?://\S+', '', text)
            text = re.sub(r'www\.\S+', '', text)
        
        # 去除邮箱
        if self.config.remove_emails:
            import re
            text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # 规范化空白
        if self.config.normalize_whitespace:
            text = ' '.join(text.split())
        
        # 长度限制
        if len(text) > self.config.max_length:
            text = text[:self.config.max_length]
        
        return text.strip()
    
    def detect_language(self, text: str) -> str:
        """检测语言"""
        try:
            from langdetect import detect
            return detect(text[:500])
        except:
            # 简单启发式判断
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            if chinese_chars > len(text) * 0.3:
                return "zh"
            return "en"


class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.preprocessor = TextPreprocessor(config)
    
    def download_hf_dataset(self, dataset_name: str) -> Iterator[Dict[str, Any]]:
        """下载 HuggingFace 数据集"""
        if not HF_AVAILABLE:
            logger.warning(f"HuggingFace datasets not available, skipping {dataset_name}")
            return
        
        logger.info(f"Downloading dataset: {dataset_name}")
        
        try:
            # 尝试加载数据集
            dataset = load_dataset(dataset_name, trust_remote_code=True)
            
            # 获取所有分割
            splits = list(dataset.keys())
            logger.info(f"Available splits: {splits}")
            
            count = 0
            for split in splits:
                for item in dataset[split]:
                    if count >= self.config.max_samples:
                        break
                    
                    # 提取文本内容
                    text = self._extract_text(item)
                    if text:
                        processed = self.preprocessor.preprocess(text)
                        if processed and len(processed) >= self.config.min_length:
                            # 语言过滤
                            if self.config.filter_language != "all":
                                lang = self.preprocessor.detect_language(processed)
                                if lang != self.config.filter_language and lang != "unknown":
                                    continue
                            
                            yield {
                                "text": processed,
                                "source": dataset_name,
                                "split": split,
                            }
                            count += 1
            
            logger.info(f"Processed {count} samples from {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {e}")
    
    def _extract_text(self, item: Dict) -> str:
        """从数据集条目中提取文本"""
        # 常见字段名
        text_fields = ['text', 'content', 'instruction', 'input', 'question', 'prompt']
        response_fields = ['output', 'response', 'answer', 'completion']
        
        # 尝试组合 instruction + output 格式
        instruction = None
        output = None
        
        for field in text_fields:
            if field in item and item[field]:
                instruction = item[field]
                break
        
        for field in response_fields:
            if field in item and item[field]:
                output = item[field]
                break
        
        # 组合文本
        if instruction and output:
            return f"{instruction}\n{output}"
        elif instruction:
            return instruction
        elif 'text' in item:
            return item['text']
        elif 'content' in item:
            return item['content']
        
        # 尝试找到任何文本字段
        for key, value in item.items():
            if isinstance(value, str) and len(value) > 50:
                return value
        
        return ""
    
    def load_local_files(self, directory: str) -> Iterator[Dict[str, Any]]:
        """加载本地文件"""
        logger.info(f"Loading local files from: {directory}")
        
        path = Path(directory)
        if not path.exists():
            logger.warning(f"Directory not found: {directory}")
            return
        
        count = 0
        for file_path in path.glob("**/*"):
            if file_path.suffix in ['.txt', '.json', '.jsonl', '.csv']:
                try:
                    if file_path.suffix == '.txt':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            processed = self.preprocessor.preprocess(text)
                            if processed:
                                yield {
                                    "text": processed,
                                    "source": str(file_path),
                                }
                                count += 1
                    
                    elif file_path.suffix in ['.json', '.jsonl']:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    item = json.loads(line.strip())
                                    text = self._extract_text(item)
                                    processed = self.preprocessor.preprocess(text)
                                    if processed:
                                        yield {
                                            "text": processed,
                                            "source": str(file_path),
                                        }
                                        count += 1
                                except json.JSONDecodeError:
                                    continue
                
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
                
                if count >= self.config.max_samples:
                    break
        
        logger.info(f"Loaded {count} samples from local files")


class DataSplitter:
    """数据划分器"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
    
    def split(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """划分数据集"""
        random.shuffle(samples)
        
        total = len(samples)
        train_end = int(total * self.config.train_ratio)
        val_end = train_end + int(total * self.config.val_ratio)
        
        return {
            "train": samples[:train_end],
            "validation": samples[train_end:val_end],
            "test": samples[val_end:],
        }


class Deduplicator:
    """去重器"""
    
    def __init__(self):
        self.seen_hashes = set()
    
    def deduplicate(self, samples: List[Dict]) -> List[Dict]:
        """去除重复样本"""
        unique = []
        for sample in samples:
            text_hash = hash(sample['text'])
            if text_hash not in self.seen_hashes:
                self.seen_hashes.add(text_hash)
                unique.append(sample)
        
        logger.info(f"Deduplicated: {len(samples)} -> {len(unique)}")
        return unique


class DataWriter:
    """数据写入器"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
    
    def write(self, split_name: str, samples: List[Dict]):
        """写入数据"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{split_name}.{self.config.output_format}"
        
        if self.config.output_format == "jsonl":
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        elif self.config.output_format == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
        
        elif self.config.output_format == "parquet":
            if HF_AVAILABLE:
                import pandas as pd
                df = pd.DataFrame(samples)
                df.to_parquet(output_file)
            else:
                logger.warning("Parquet format requires pandas/pyarrow")
        
        logger.info(f"Wrote {len(samples)} samples to {output_file}")
    
    def write_stats(self, splits: Dict[str, List[Dict]]):
        """写入统计信息"""
        output_dir = Path(self.config.output_dir)
        stats_file = output_dir / "stats.json"
        
        stats = {
            "total_samples": sum(len(s) for s in splits.values()),
            "splits": {
                name: {
                    "count": len(samples),
                    "avg_length": sum(len(s['text']) for s in samples) / len(samples) if samples else 0,
                }
                for name, samples in splits.items()
            },
            "config": {
                "datasets": self.config.datasets,
                "max_samples": self.config.max_samples,
                "min_length": self.config.min_length,
                "max_length": self.config.max_length,
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Wrote stats to {stats_file}")


def prepare_data(config: DataPrepConfig):
    """准备训练数据"""
    logger.info("Starting data preparation...")
    
    downloader = DatasetDownloader(config)
    deduplicator = Deduplicator()
    splitter = DataSplitter(config)
    writer = DataWriter(config)
    
    # 收集所有样本
    all_samples = []
    
    # 下载 HuggingFace 数据集
    for dataset_name in config.datasets:
        for sample in downloader.download_hf_dataset(dataset_name):
            all_samples.append(sample)
    
    # 去重
    if config.deduplicate:
        all_samples = deduplicator.deduplicate(all_samples)
    
    # 划分
    splits = splitter.split(all_samples)
    
    # 写入
    for split_name, samples in splits.items():
        writer.write(split_name, samples)
    
    # 写入统计
    writer.write_stats(splits)
    
    logger.info(f"Data preparation complete. Total: {len(all_samples)} samples")
    return splits


def create_sample_dataset(output_dir: str = "data/sample", num_samples: int = 1000):
    """创建示例数据集（用于测试）"""
    logger.info(f"Creating sample dataset with {num_samples} samples...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 生成示例对话数据
    templates = [
        ("用户：{}\n助手：{}", "问答"),
        ("问题：{}\n答案：{}", "问答"),
        ("指令：{}\n输出：{}", "指令"),
        ("输入：{}\n输出：{}", "转换"),
    ]
    
    sample_questions = [
        "什么是人工智能？",
        "请解释机器学习的基本概念。",
        "深度学习和机器学习有什么区别？",
        "什么是神经网络？",
        "自然语言处理有哪些应用？",
        "请介绍一下Transformer架构。",
        "什么是注意力机制？",
        "BERT和GPT有什么区别？",
        "如何评估语言模型的性能？",
        "什么是RLHF？",
    ]
    
    sample_answers = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子领域，它使计算机系统能够从数据中学习并改进，而无需显式编程。",
        "深度学习是机器学习的一个子集，使用多层神经网络来处理复杂的数据模式。",
        "神经网络是一种受人脑启发的计算模型，由相互连接的节点层组成。",
        "自然语言处理应用于机器翻译、情感分析、文本摘要、问答系统等领域。",
        "Transformer是一种基于自注意力机制的神经网络架构，广泛应用于NLP任务。",
        "注意力机制允许模型在处理序列时关注最相关的部分，提高了信息处理效率。",
        "BERT使用双向编码器，适合理解任务；GPT使用单向解码器，适合生成任务。",
        "常用指标包括困惑度、BLEU分数、准确率、F1分数等。",
        "RLHF是从人类反馈中学习强化学习，用于使模型输出更符合人类偏好。",
    ]
    
    samples = []
    for i in range(num_samples):
        q_idx = i % len(sample_questions)
        template, source = templates[i % len(templates)]
        
        sample = {
            "text": template.format(sample_questions[q_idx], sample_answers[q_idx]),
            "source": source,
            "id": i,
        }
        samples.append(sample)
    
    # 划分
    train_end = int(num_samples * 0.9)
    
    splits = {
        "train": samples[:train_end],
        "validation": samples[train_end:],
    }
    
    # 写入
    for split_name, split_samples in splits.items():
        output_file = output_path / f"{split_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in split_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Wrote {len(split_samples)} samples to {output_file}")
    
    # 写入统计
    stats_file = output_path / "stats.json"
    stats = {
        "total_samples": num_samples,
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "avg_length": sum(len(s['text']) for s in samples) / len(samples),
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Sample dataset created at {output_path}")
    return splits


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据准备脚本")
    parser.add_argument("--output-dir", default="data/prepared", help="输出目录")
    parser.add_argument("--max-samples", type=int, default=100000, help="最大样本数")
    parser.add_argument("--min-length", type=int, default=50, help="最小文本长度")
    parser.add_argument("--max-length", type=int, default=2048, help="最大文本长度")
    parser.add_argument("--language", default="zh", choices=["zh", "en", "all"], help="语言过滤")
    parser.add_argument("--sample", action="store_true", help="创建示例数据集")
    parser.add_argument("--sample-count", type=int, default=1000, help="示例数据集大小")
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_dataset(args.output_dir, args.sample_count)
    else:
        config = DataPrepConfig(
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            min_length=args.min_length,
            max_length=args.max_length,
            filter_language=args.language,
        )
        prepare_data(config)


if __name__ == "__main__":
    main()