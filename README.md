# Nexus-7B: 企业级大语言模型训练框架

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-2.0.0-blue.svg)]()

**Nexus-7B** 是一个功能完备的企业级大语言模型训练框架，支持从预训练到RLHF微调、Agent开发、模型压缩、增量学习等全生命周期管理。

## 🌟 核心特性

### 基础能力
- **Transformer 架构**: Decoder-only结构，支持GQA、RoPE、SwiGLU
- **Flash Attention 2**: 高效注意力计算，显著降低显存占用
- **分布式训练**: DeepSpeed ZeRO-3、FSDP、DDP 多节点支持
- **混合精度**: FP16/BF16 自动混合精度训练

### RLHF 训练
- **Reward Model**: 基于偏好数据训练奖励模型
- **PPO 训练**: 近端策略优化，实现人类反馈强化学习
- **KL 散度约束**: 自适应系数调整，稳定训练过程

### Agent 工具链
- **Tool Use**: 函数调用框架，支持外部工具集成
- **ReAct 推理**: 思考-行动-观察循环
- **多Agent协作**: 任务委托与协调
- **内置工具**: 计算器、搜索、知识库等

### 模型压缩
- **知识蒸馏**: 大模型到小模型知识迁移
- **结构化剪枝**: 注意力头、FFN层剪枝
- **量化**: INT8/INT4 动态/静态量化
- **压缩比**: 可达 4-16x

### 增量学习
- **Experience Replay**: 经验回放防止遗忘
- **EWC 正则化**: Elastic Weight Consolidation
- **Adapter**: 轻量级增量模块，仅训练 0.1% 参数
- **持续学习**: 多任务顺序训练支持

## 📦 安装

```bash
# 完整安装
pip install nexus-llm[all]

# 基础安装
pip install nexus-llm

# 开发安装
pip install -e ".[dev]"
```

## 🚀 快速开始

### 1. 模型训练
```python
from nexus_llm import NexusConfig, NexusForCausalLM, Trainer

config = NexusConfig(
    hidden_size=4096,
    num_hidden_layers=32,
)

model = NexusForCausalLM(config)
trainer = Trainer(model, tokenizer, train_loader, eval_loader)
trainer.train()
```

### 2. RLHF 微调
```python
from nexus_llm import create_rlhf_pipeline, RLHFConfig

ppo_trainer, reward_trainer = create_rlhf_pipeline(
    policy_model=model,
    tokenizer=tokenizer,
    config=RLHFConfig(),
)

# 训练 Reward Model
reward_trainer.train(preference_data)

# PPO 训练
ppo_trainer.train(prompts)
```

### 3. Agent 开发
```python
from nexus_llm import ReActAgent, ToolRegistry, Calculator

registry = ToolRegistry()
registry.register(Calculator())

agent = ReActAgent(model, tokenizer, registry)
result = agent.run_sync("请计算 25 * 4 + 10")
```

### 4. 模型压缩
```python
from nexus_llm import ModelCompressor, QuantizationConfig

compressor = ModelCompressor(model, tokenizer)
compressed = compressor.compress(
    method="quantization",
    config=QuantizationConfig(bits=8)
)
```

## 📁 项目结构

```
nexus-llm/
├── src/nexus_llm/
│   ├── model/           # Transformer模型架构
│   ├── data/            # 数据处理和预处理
│   ├── training/        # 训练系统
│   ├── serving/         # 推理部署
│   ├── enterprise/      # 企业特性
│   ├── evaluation/      # 评估基准
│   ├── rlhf/           # RLHF训练模块 ⭐ NEW
│   ├── agent/          # Agent工具链 ⭐ NEW
│   ├── compression/     # 模型压缩 ⭐ NEW
│   └── incremental/     # 增量学习 ⭐ NEW
├── configs/             # 配置文件
├── scripts/            # 启动脚本
├── examples/           # 使用示例
├── tests/              # 测试
└── README.md
```

## 🛠️ 模块详解

### RLHF 模块
| 组件 | 功能 |
|------|------|
| RewardModel | 预测人类偏好评分 |
| PPOTrainer | 近端策略优化训练 |
| RewardModelTrainer | 偏好数据训练 |

### Agent 模块
| 组件 | 功能 |
|------|------|
| ReActAgent | 思考-行动-观察循环 |
| FunctionCallingAgent | 函数调用 |
| MultiAgentCoordinator | 多Agent协作 |
| ToolRegistry | 工具注册管理 |

### Compression 模块
| 组件 | 功能 |
|------|------|
| KnowledgeDistiller | 知识蒸馏 |
| StructuredPruner | 结构化剪枝 |
| ModelCompressor | 量化(Dynamic/Static/QAT) |

### Incremental 模块
| 组件 | 功能 |
|------|------|
| ExperienceReplayBuffer | 经验回放 |
| EWCRegularizer | EWC正则化 |
| AdapterModel | Adapter增量 |

## 📊 性能指标

### 训练性能
| 配置 | 显存 | 速度 |
|------|------|------|
| BF16 单卡 | ~24GB | 0.5 samples/s |
| 8卡 ZeRO-3 | ~8GB/卡 | 4 samples/s |
| LoRA 微调 | ~12GB | 1 sample/s |

### 推理性能
| 精度 | 显存 | 速度 |
|------|------|------|
| BF16 | ~16GB | 50 tokens/s |
| INT8 | ~8GB | 60 tokens/s |
| INT4 | ~5GB | 70 tokens/s |

## 🔒 企业级特性

- **监控**: Prometheus + Grafana 指标收集
- **安全**: API密钥、权限控制、速率限制
- **审计**: 完整操作日志记录
- **高可用**: 检查点复制、自动恢复
- **合规**: GDPR、SOC2 数据保护

## 📚 文档

- [快速开始指南](docs/getting_started.md)
- [RLHF 训练教程](docs/rlhf_guide.md)
- [Agent 开发指南](docs/agent_guide.md)
- [模型压缩指南](docs/compression_guide.md)
- [API 参考文档](docs/api_reference.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

Apache 2.0 License

## 🙏 致谢

- PyTorch
- Hugging Face Transformers
- DeepSpeed
- Flash Attention
