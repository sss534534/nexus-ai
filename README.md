# Nexus-7B: 企业级垂直领域大语言模型训练框架

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)

Nexus-7B 是一个企业级的 7B 参数大语言模型训练框架，专为垂直领域应用设计。框架提供了完整的训练、微调、部署和推理能力，达到商业化应用水平。

## 🌟 核心特性

### 模型架构
- **Decoder-only Transformer**: 采用先进的解码器架构
- **Group Query Attention (GQA)**: 支持多组查询注意力，提升推理效率
- **Rotary Position Embedding (RoPE)**: 动态位置编码，支持长序列
- **SwiGLU 激活函数**: 提升模型表达能力
- **Flash Attention 2**: 高效注意力计算，显著降低显存占用

### 训练系统
- **分布式训练**: 支持 DeepSpeed ZeRO-3、FSDP、DDP
- **混合精度训练**: FP16/BF16 自动混合精度
- **梯度检查点**: 显存优化，支持大模型训练
- **学习率调度**: Cosine/Linear/Warmup 调度策略
- **LoRA/QLoRA**: 低秩适配微调，降低训练成本

### 数据处理
- **SentencePiece/BPE 分词**: 支持多种分词算法
- **高效数据加载**: 流式数据加载，支持大规模数据集
- **垂直领域模板**: 金融、医疗、法律、代码等领域模板
- **数据预处理**: 自动清洗、分割、格式化

### 企业级特性
- **Prometheus 监控**: 完整的指标收集和暴露
- **审计日志**: 符合企业合规要求
- **API 安全**: 密钥认证、权限控制、速率限制
- **高可用**: 检查点复制、自动恢复、健康检查
- **加密存储**: 敏感数据加密保护

### 推理部署
- **量化推理**: INT8/INT4 量化，降低部署成本
- **流式输出**: 支持实时流式生成
- **批量推理**: 高效批量处理
- **KV Cache**: 推理缓存优化
- **FastAPI 服务**: 生产级 API 服务

## 📦 安装

### 基础安装

```bash
pip install nexus-llm
```

### 完整安装（推荐）

```bash
pip install nexus-llm[all]
```

### 开发安装

```bash
git clone https://github.com/nexus-ai/nexus-llm.git
cd nexus-llm
pip install -e ".[dev]"
```

### 依赖说明

| 组件 | 必需依赖 | 可选依赖 |
|------|----------|----------|
| 基础训练 | torch, transformers, accelerate | - |
| 高效训练 | - | deepspeed, flash-attn |
| 分词器 | tokenizers | sentencepiece |
| 监控 | - | wandb, prometheus-client |
| 量化 | - | bitsandbytes, auto-gptq |
| 服务 | - | fastapi, uvicorn |

## 🚀 快速开始

### 1. 训练模型

```bash
# 单机训练
nexus-train --model-config configs/model_config.yaml \
             --training-config configs/training_config.yaml \
             --data-config configs/data_config.yaml

# 分布式训练（8卡）
torchrun --nproc_per_node=8 nexus-train \
         --model-config configs/model_config.yaml \
         --distributed
```

### 2. 启动推理服务

```bash
nexus-serve --model-path checkpoints/best \
            --tokenizer-path tokenizer/nexus_tokenizer.model \
            --port 8000
```

### 3. API 调用

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "请解释什么是深度学习？",
        "max_new_tokens": 512,
        "temperature": 0.7,
    }
)

print(response.json()["text"])
```

## 📖 详细使用

### Python API 使用

```python
from nexus_llm import NexusModel, NexusConfig, NexusTokenizer
from nexus_llm.training import Trainer, TrainingConfig
from nexus_llm.serving import InferenceEngine

# 创建模型配置
config = NexusConfig(
    hidden_size=4096,
    num_attention_heads=32,
    num_hidden_layers=32,
    vocab_size=152064,
    max_position_embeddings=8192,
)

# 创建模型
model = NexusModel(config)

# 加载分词器
tokenizer = NexusTokenizer.from_pretrained("tokenizer/nexus_tokenizer.model")

# 创建推理引擎
engine = InferenceEngine(model, tokenizer)

# 生成文本
response = engine.generate(
    prompt="什么是人工智能？",
    max_new_tokens=256,
    temperature=0.7,
)

print(response["text"])
```

### 训练流程

```python
from nexus_llm import NexusForCausalLM, NexusConfig
from nexus_llm.data import DataProcessor, DataConfig, NexusTokenizer
from nexus_llm.training import Trainer, TrainingConfig

# 配置
model_config = NexusConfig()
data_config = DataConfig(
    train_file="data/train.jsonl",
    valid_file="data/valid.jsonl",
    max_seq_length=8192,
)
training_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=4,
    max_steps=100000,
    output_dir="outputs/nexus-7b",
)

# 创建组件
model = NexusForCausalLM(model_config)
tokenizer = NexusTokenizer(data_config)
processor = DataProcessor(data_config, tokenizer)

# 创建数据集
datasets = processor.create_datasets()
dataloaders = processor.create_dataloaders(datasets)

# 创建训练器
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataloader=dataloaders['train'],
    eval_dataloader=dataloaders['valid'],
    config=training_config,
)

# 开始训练
trainer.train()
```

### LoRA 微调

```python
from nexus_llm.training import FineTuningTrainer, TrainingConfig

lora_config = {
    "r": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "dropout": 0.05,
}

trainer = FineTuningTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataloader=dataloaders['train'],
    config=training_config,
    lora_config=lora_config,
)

trainer.train()
```

### 垂直领域定制

```python
from nexus_llm.data import VerticalDomainProcessor

# 金融领域
processor = VerticalDomainProcessor(
    config=data_config,
    tokenizer=tokenizer,
    domain="finance",
)

# 处理金融数据
processor.preprocess_raw_data(
    raw_dir="data/finance_raw",
    output_dir="data/finance_processed",
)
```

### 企业级部署

```python
from nexus_llm.enterprise import EnterpriseManager, EnterpriseConfig
from nexus_llm.serving import create_model_server

# 企业配置
enterprise_config = EnterpriseConfig(
    monitoring_enabled=True,
    prometheus_port=9090,
    encryption_enabled=True,
    audit_logging=True,
    access_control=True,
)

# 创建服务器
server = create_model_server(
    model_path="checkpoints/best",
    tokenizer_path="tokenizer/nexus_tokenizer.model",
    enterprise_config=enterprise_config,
)

# 启动服务
server.run(host="0.0.0.0", port=8000)
```

## 📁 项目结构

```
nexus-llm/
├── src/nexus_llm/
│   ├── model/           # 模型架构
│   │   ├── transformer.py
│   │   ├── attention.py
│   │   ├── embedding.py
│   │   └── layers.py
│   ├── data/            # 数据处理
│   │   ├── tokenizer.py
│   │   ├── dataset.py
│   │   └── processor.py
│   ├── training/        # 训练系统
│   │   ├── trainer.py
│   │   ├── distributed.py
│   │   ├── optimizer.py
│   │   └── scheduler.py
│   ├── serving/         # 推理部署
│   │   ├── engine.py
│   │   ├── server.py
│   │   └── quantization.py
│   ├── enterprise/      # 企业特性
│   │   ├── monitoring.py
│   │   ├── security.py
│   │   ├── audit.py
│   │   └── ha.py
│   └── utils/           # 工具函数
├── configs/             # 配置文件
│   ├── model_config.yaml
│   ├── training_config.yaml
│   ├── deepspeed/
│   └── enterprise/
├── scripts/             # 脚本
│   ├── train.sh
│   ├── serve.sh
│   └── eval.sh
├── docs/                # 文档
├── tests/               # 测试
├── examples/            # 示例
└── pyproject.toml
```

## ⚙️ 配置说明

### 模型配置 (model_config.yaml)

```yaml
model:
  hidden_size: 4096
  num_attention_heads: 32
  num_hidden_layers: 32
  vocab_size: 152064
  max_position_embeddings: 8192
  rope_theta: 10000.0
  use_flash_attention: true
```

### 训练配置 (training_config.yaml)

```yaml
training:
  learning_rate: 1.0e-4
  batch_size: 4
  gradient_accumulation_steps: 8
  max_steps: 100000
  precision: bf16
  gradient_checkpointing: true
  
distributed:
  strategy: deepspeed
  deepspeed_config: configs/deepspeed/zero3.json
```

### DeepSpeed 配置

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  },
  "bf16": {"enabled": true}
}
```

## 📊 性能指标

### 训练性能

| 配置 | 显存占用 | 训练速度 |
|------|----------|----------|
| 单卡 BF16 | ~24GB | ~0.5 samples/s |
| 8卡 DeepSpeed ZeRO-3 | ~8GB/卡 | ~4 samples/s |
| LoRA 微调 | ~12GB | ~1 sample/s |

### 推理性能

| 配置 | 显存占用 | 推理速度 |
|------|----------|----------|
| BF16 | ~16GB | ~50 tokens/s |
| INT8 量化 | ~8GB | ~60 tokens/s |
| INT4 量化 | ~5GB | ~70 tokens/s |

## 🔒 安全特性

- **API 密钥认证**: 所有 API 调用需要有效密钥
- **权限控制**: 基于角色的访问控制
- **速率限制**: 防止滥用和 DDoS
- **数据加密**: AES-256 加密存储
- **审计日志**: 完整操作记录
- **合规支持**: GDPR、SOC2 等

## 🧪 测试

```bash
# 运行单元测试
pytest tests/

# 运行覆盖率测试
pytest --cov=nexus_llm tests/

# 运行集成测试
pytest tests/integration/
```

## 📚 文档

- [架构设计文档](docs/architecture.md)
- [训练指南](docs/training_guide.md)
- [部署指南](docs/deployment_guide.md)
- [API 文档](docs/api_reference.md)
- [企业特性](docs/enterprise_features.md)

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 Apache 2.0 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Hugging Face](https://huggingface.co/) - Transformers 库
- [DeepSpeed](https://www.deepspeed.ai/) - 分布式训练
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - 高效注意力

## 📧 联系方式

- 项目主页: https://github.com/nexus-ai/nexus-llm
- 文档站点: https://nexus-llm.readthedocs.io
- 问题反馈: https://github.com/nexus-ai/nexus-llm/issues
- 邮件: team@nexus-ai.com