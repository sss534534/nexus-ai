#!/usr/bin/env python3
"""
Tokenizer 训练脚本 - 训练 SentencePiece 分词器
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional

try:
    import sentencepiece as spm
    SP_AVAILABLE = True
except ImportError:
    SP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_tokenizer(
    input_file: str,
    output_dir: str,
    vocab_size: int = 32000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    max_sentence_length: int = 2048,
    pad_id: int = 0,
    unk_id: int = 1,
    bos_id: int = 2,
    eos_id: int = 3,
):
    """
    训练 SentencePiece 分词器
    
    Args:
        input_file: 输入文本文件路径
        output_dir: 输出目录
        vocab_size: 词表大小
        model_type: 模型类型 (bpe, unigram, char, word)
        character_coverage: 字符覆盖率
        max_sentence_length: 最大句子长度
    """
    if not SP_AVAILABLE:
        logger.error("SentencePiece not installed. Run: pip install sentencepiece")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_prefix = output_path / "nexus_tokenizer"
    
    logger.info(f"Training tokenizer with vocab_size={vocab_size}")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {model_prefix}")
    
    # 训练参数
    train_args = [
        f"--input={input_file}",
        f"--model_prefix={str(model_prefix)}",
        f"--vocab_size={vocab_size}",
        f"--model_type={model_type}",
        f"--character_coverage={character_coverage}",
        f"--max_sentence_length={max_sentence_length}",
        f"--pad_id={pad_id}",
        f"--unk_id={unk_id}",
        f"--bos_id={bos_id}",
        f"--eos_id={eos_id}",
        "--byte_fallback=true",  # 支持未知字符
        "--split_by_unicode_script=true",
        "--split_by_number=true",
        "--split_by_whitespace=true",
        "--treat_whitespace_as_suffix=false",
        "--allow_whitespace_only_pieces=true",
        "--remove_extra_whitespaces=false",
        "--normalization_rule_name=nmt_nfkc_cf",  # NFKC规范化
    ]
    
    # 训练
    spm.SentencePieceTrainer.train(' '.join(train_args))
    
    logger.info(f"Tokenizer trained successfully!")
    logger.info(f"Model file: {model_prefix}.model")
    logger.info(f"Vocab file: {model_prefix}.vocab")
    
    # 验证
    sp = spm.SentenceProcessor()
    sp.load(str(model_prefix) + ".model")
    
    # 测试编码
    test_text = "这是一个测试句子。This is a test sentence."
    encoded = sp.encode(test_text, out_type=int)
    decoded = sp.decode(encoded)
    
    logger.info(f"Test encoding: '{test_text}' -> {encoded[:10]}...")
    logger.info(f"Test decoding: '{decoded}'")
    
    return str(model_prefix) + ".model"


def create_sample_corpus(output_file: str, num_lines: int = 10000):
    """创建示例语料库用于训练 tokenizer"""
    logger.info(f"Creating sample corpus with {num_lines} lines...")
    
    # 中文示例
    chinese_samples = [
        "人工智能是计算机科学的一个分支。",
        "机器学习使计算机能够从数据中学习。",
        "深度学习使用多层神经网络处理复杂数据。",
        "自然语言处理是人工智能的重要应用领域。",
        "Transformer架构在自然语言处理中广泛应用。",
        "注意力机制提高了模型处理序列的能力。",
        "BERT和GPT是两种重要的预训练语言模型。",
        "RLHF是从人类反馈中学习的方法。",
        "模型压缩可以减少模型大小提高推理效率。",
        "增量学习允许模型持续学习新知识。",
    ]
    
    # 英文示例
    english_samples = [
        "Artificial intelligence is a branch of computer science.",
        "Machine learning enables computers to learn from data.",
        "Deep learning uses multi-layer neural networks.",
        "Natural language processing is an important AI application.",
        "Transformer architecture is widely used in NLP.",
        "Attention mechanism improves sequence processing.",
        "BERT and GPT are important pre-trained language models.",
        "RLHF is a method for learning from human feedback.",
        "Model compression reduces model size and improves efficiency.",
        "Incremental learning allows models to learn continuously.",
    ]
    
    # 代码示例
    code_samples = [
        "def train_model(model, data, epochs=10):",
        "    optimizer = torch.optim.Adam(model.parameters())",
        "    for epoch in range(epochs):",
        "        loss = model.forward(data)",
        "        optimizer.step()",
        "class NeuralNetwork(nn.Module):",
        "    def __init__(self, hidden_size=256):",
        "        super().__init__()",
        "        self.linear = nn.Linear(hidden_size, hidden_size)",
        "    def forward(self, x):",
        "        return self.linear(x)",
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_lines):
            # 随机选择样本类型
            if i % 3 == 0:
                sample = chinese_samples[i % len(chinese_samples)]
            elif i % 3 == 1:
                sample = english_samples[i % len(english_samples)]
            else:
                sample = code_samples[i % len(code_samples)]
            
            f.write(sample + '\n')
    
    logger.info(f"Sample corpus created: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Tokenizer 训练脚本")
    parser.add_argument("--input", required=False, help="输入文本文件")
    parser.add_argument("--output-dir", default="tokenizer", help="输出目录")
    parser.add_argument("--vocab-size", type=int, default=32000, help="词表大小")
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"], help="模型类型")
    parser.add_argument("--sample", action="store_true", help="创建示例语料库")
    parser.add_argument("--sample-lines", type=int, default=10000, help="示例语料库行数")
    
    args = parser.parse_args()
    
    if args.sample:
        # 创建示例语料库
        corpus_file = Path(args.output_dir) / "sample_corpus.txt"
        create_sample_corpus(str(corpus_file), args.sample_lines)
        input_file = str(corpus_file)
    elif args.input:
        input_file = args.input
    else:
        logger.error("Either --input or --sample must be specified")
        return
    
    train_tokenizer(
        input_file=input_file,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()