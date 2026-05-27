# Nexus-LLM 快速入门指南

> Nexus-LLM 是一个高性能、可扩展的大语言模型训练与推理框架，支持从预训练到 RLHF 的完整流程，并提供 OpenAI 兼容的推理服务接口。

---

## 目录

- [环境要求](#环境要求)
- [安装](#安装)
  - [通过 pip 安装](#通过-pip-安装)
  - [从源码安装](#从源码安装)
  - [通过 Docker 安装](#通过-docker-安装)
- [快速上手：模型训练](#快速上手模型训练)
- [快速上手：模型推理](#快速上手模型推理)
- [命令行工具](#命令行工具)
  - [nexus-train](#nexus-train)
  - [nexus-serve](#nexus-serve)
  - [nexus-eval](#nexus-eval)
- [常见问题](#常见问题)

---

## 环境要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| Python | 3.9+ | 3.10 或 3.11 |
| PyTorch | 2.0+ | 2.1+ |
| CUDA | 11.8+ | 12.1+ |
| GPU 显存 | 24 GB（7B 模型） | 80 GB（70B 模型） |
| 系统内存 | 32 GB | 128 GB+ |
| 磁盘空间 | 50 GB | 500 GB+ SSD |

> **提示：** 如果仅进行推理（不训练），GPU 显存要求可以降低。使用量化技术后，7B 模型仅需约 8 GB 显存。

---

## 安装

### 通过 pip 安装

```bash
# 创建虚拟环境（推荐）
python -m venv nexus-env
source nexus-env/bin/activate

# 安装核心包
pip install nexus-llm

# 安装训练依赖
pip install "nexus-llm[train]"

# 安装推理服务依赖
pip install "nexus-llm[serve]"

# 安装 RLHF 依赖
pip install "nexus-llm[rlhf]"

# 安装 Agent 依赖
pip install "nexus-llm[agent]"

# 安装全部依赖（开发推荐）
pip install "nexus-llm[all]"
```

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/nexus-ai/nexus-llm.git
cd nexus-llm

# 安装开发依赖
pip install -e ".[dev]"

# 验证安装
python -c "import nexus_llm; print(nexus_llm.__version__)"
```

### 通过 Docker 安装

```bash
# 拉取官方镜像
docker pull nexusai/nexus-llm:latest

# 运行推理服务容器
docker run -d \
  --name nexus-serve \
  --gpus all \
  -p 8000:8000 \
  -v ~/.nexus/models:/root/.nexus/models \
  nexusai/nexus-llm:latest \
  nexus-serve --model-path /root/.nexus/models/nexus-7b \
               --host 0.0.0.0 \
               --port 8000

# 运行训练容器
docker run -it --rm \
  --gpus all \
  -v ~/data:/data \
  -v ~/.nexus:/root/.nexus \
  nexusai/nexus-llm:latest \
  nexus-train --config /data/train_config.yaml
```

---

## 快速上手：模型训练

以下示例展示如何使用 Nexus-LLM 对模型进行微调训练。

### 1. 准备训练数据

训练数据支持 JSONL 格式，每行一个样本：

```jsonl
{"instruction": "请解释什么是机器学习", "output": "机器学习是人工智能的一个分支..."}
{"instruction": "用Python写一个快速排序", "output": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    ..."}
```

### 2. 编写训练脚本

```python
"""basic_finetune.py - 基础微调训练示例"""
import torch
from nexus_llm import NexusConfig, NexusForCausalLM
from nexus_llm.training import Trainer, TrainingConfig

# 1. 加载模型配置
config = NexusConfig(
    model_name="nexus-7b",
    hidden_size=4096,
    num_attention_heads=32,
    num_hidden_layers=32,
    intermediate_size=11008,
    max_position_embeddings=4096,
    vocab_size=32000,
)

# 2. 初始化模型
model = NexusForCausalLM(config)

# 3. 配置训练参数
training_config = TrainingConfig(
    output_dir="./output/nexus-7b-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    weight_decay=0.01,
    fp16=True,                          # 启用混合精度训练
    gradient_checkpointing=True,        # 节省显存
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    dataloader_num_workers=4,
)

# 4. 初始化训练器
trainer = Trainer(
    model=model,
    config=training_config,
    train_file="./data/train.jsonl",
    eval_file="./data/eval.jsonl",
)

# 5. 开始训练
trainer.train()

# 6. 保存模型
trainer.save_model("./output/nexus-7b-finetuned")
print("训练完成！模型已保存到 ./output/nexus-7b-finetuned")
```

### 3. 运行训练

```bash
# 运行微调脚本
python basic_finetune.py

# 或使用命令行工具
nexus-train \
  --model-path nexus-ai/nexus-7b \
  --train-file ./data/train.jsonl \
  --eval-file ./data/eval.jsonl \
  --output-dir ./output/nexus-7b-finetuned \
  --num-epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-5 \
  --fp16
```

---

## 快速上手：模型推理

### 1. Python API 推理

```python
"""basic_inference.py - 基础推理示例"""
from nexus_llm import NexusForCausalLM, NexusTokenizer
from nexus_llm.serving import InferenceEngine

# 1. 加载模型和分词器
model = NexusForCausalLM.from_pretrained("nexus-ai/nexus-7b-finetuned")
tokenizer = NexusTokenizer.from_pretrained("nexus-ai/nexus-7b-finetuned")

# 2. 创建推理引擎
engine = InferenceEngine(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
)

# 3. 简单文本生成
prompt = "请用简洁的语言解释量子计算的基本原理："
response = engine.generate(prompt)
print(f"问题: {prompt}")
print(f"回答: {response}")

# 4. 对话模式
messages = [
    {"role": "system", "content": "你是一个有帮助的AI助手。"},
    {"role": "user", "content": "什么是深度学习？"},
]
response = engine.chat(messages)
print(f"回答: {response}")

# 5. 流式输出
print("流式输出：")
for token in engine.generate_stream("请写一首关于春天的诗："):
    print(token, end="", flush=True)
print()
```

### 2. 启动推理服务

```bash
# 启动 OpenAI 兼容的推理服务
nexus-serve \
  --model-path nexus-ai/nexus-7b-finetuned \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096

# 后台运行
nohup nexus-serve \
  --model-path nexus-ai/nexus-7b-finetuned \
  --port 8000 \
  > nexus-serve.log 2>&1 &
```

### 3. 使用 OpenAI 兼容 API 调用

```python
"""api_inference.py - 通过 HTTP API 调用推理服务"""
import openai

# 配置客户端（指向本地 Nexus-LLM 服务）
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # Nexus-LLM 默认不需要 API Key
)

# Chat Completions
response = client.chat.completions.create(
    model="nexus-7b-finetuned",
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用Python实现一个LRU缓存"},
    ],
    temperature=0.7,
    max_tokens=2048,
    stream=True,
)

# 流式读取响应
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

# 查看可用模型列表
models = client.models.list()
for model in models.data:
    print(f"模型: {model.id}")
```

### 4. 使用 curl 调用

```bash
# Chat Completions
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nexus-7b-finetuned",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'

# 查看模型列表
curl http://localhost:8000/v1/models
```

---

## 命令行工具

Nexus-LLM 提供三个核心命令行工具，覆盖训练、推理服务和评估的完整工作流。

### nexus-train

模型训练命令行工具，支持预训练、微调和 RLHF 训练。

```bash
# 基础微调
nexus-train \
  --model-path nexus-ai/nexus-7b \
  --train-file ./data/train.jsonl \
  --output-dir ./output \
  --num-epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-5

# 使用配置文件训练
nexus-train --config ./configs/finetune.yaml

# 分布式训练（多 GPU）
nexus-train \
  --model-path nexus-ai/nexus-7b \
  --train-file ./data/train.jsonl \
  --output-dir ./output \
  --num-gpus 4 \
  --strategy deepspeed_stage3

# LoRA 微调
nexus-train \
  --model-path nexus-ai/nexus-7b \
  --train-file ./data/train.jsonl \
  --output-dir ./output \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,v_proj,k_proj,o_proj
```

**完整参数列表：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-path` | str | 必填 | 模型路径或 HuggingFace 模型 ID |
| `--config` | str | None | 训练配置文件路径（YAML） |
| `--train-file` | str | 必填 | 训练数据文件路径 |
| `--eval-file` | str | None | 评估数据文件路径 |
| `--output-dir` | str | `./output` | 输出目录 |
| `--num-epochs` | int | 3 | 训练轮数 |
| `--batch-size` | int | 8 | 每设备批次大小 |
| `--gradient-accumulation-steps` | int | 1 | 梯度累积步数 |
| `--learning-rate` | float | 2e-5 | 学习率 |
| `--warmup-ratio` | float | 0.03 | 预热比例 |
| `--weight-decay` | float | 0.01 | 权重衰减 |
| `--max-seq-length` | int | 2048 | 最大序列长度 |
| `--fp16` | flag | False | 启用 FP16 混合精度 |
| `--bf16` | flag | False | 启用 BF16 混合精度 |
| `--num-gpus` | int | 1 | 使用的 GPU 数量 |
| `--strategy` | str | `ddp` | 分布式策略（ddp/deepspeed_stage2/deepspeed_stage3） |
| `--lora-r` | int | None | LoRA 秩（启用 LoRA 微调） |
| `--lora-alpha` | int | None | LoRA 缩放因子 |
| `--lora-dropout` | float | 0.05 | LoRA Dropout |
| `--lora-target-modules` | str | None | LoRA 目标模块（逗号分隔） |
| `--resume-from` | str | None | 从检查点恢复训练 |
| `--seed` | int | 42 | 随机种子 |

### nexus-serve

模型推理服务命令行工具，提供 OpenAI 兼容的 HTTP API。

```bash
# 基础启动
nexus-serve \
  --model-path nexus-ai/nexus-7b \
  --host 0.0.0.0 \
  --port 8000

# 高级配置
nexus-serve \
  --model-path nexus-ai/nexus-7b \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --dtype float16 \
  --quantization awq \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1

# 使用量化模型
nexus-serve \
  --model-path nexus-ai/nexus-7b-awq \
  --quantization awq \
  --port 8000

# 启用 SSL
nexus-serve \
  --model-path nexus-ai/nexus-7b \
  --ssl-certfile /path/to/cert.pem \
  --ssl-keyfile /path/to/key.pem \
  --port 8443
```

**完整参数列表：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-path` | str | 必填 | 模型路径或 HuggingFace 模型 ID |
| `--host` | str | `0.0.0.0` | 监听地址 |
| `--port` | int | `8000` | 监听端口 |
| `--gpu-memory-utilization` | float | 0.9 | GPU 显存利用率 |
| `--max-model-len` | int | 4096 | 模型最大上下文长度 |
| `--max-num-seqs` | int | 256 | 最大并发序列数 |
| `--dtype` | str | `auto` | 数据类型（auto/float16/bfloat16/float32） |
| `--quantization` | str | None | 量化方式（awq/gptq/squeezellm/fp8） |
| `--tensor-parallel-size` | int | 1 | 张量并行大小 |
| `--pipeline-parallel-size` | int | 1 | 流水线并行大小 |
| `--ssl-certfile` | str | None | SSL 证书文件路径 |
| `--ssl-keyfile` | str | None | SSL 密钥文件路径 |
| `--api-key` | str | None | API 密钥（为空则不鉴权） |
| `--chat-template` | str | None | 聊天模板路径 |
| `--served-model-name` | str | None | 服务对外暴露的模型名称 |

### nexus-eval

模型评估命令行工具，支持多种基准测试。

```bash
# 运行全部基准测试
nexus-eval \
  --model-path ./output/nexus-7b-finetuned \
  --tasks all \
  --output-dir ./eval_results

# 运行指定基准测试
nexus-eval \
  --model-path ./output/nexus-7b-finetuned \
  --tasks mmlu,ceval,humaneval \
  --num-fewshot 5 \
  --batch-size 8 \
  --output-dir ./eval_results

# 使用配置文件
nexus-eval --config ./configs/eval.yaml

# 生成评估报告
nexus-eval \
  --model-path ./output/nexus-7b-finetuned \
  --tasks all \
  --output-dir ./eval_results \
  --format markdown
```

**完整参数列表：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-path` | str | 必填 | 模型路径 |
| `--config` | str | None | 评估配置文件路径 |
| `--tasks` | str | `all` | 评估任务（逗号分隔或 all） |
| `--num-fewshot` | int | 0 | Few-shot 示例数量 |
| `--batch-size` | int | 8 | 批次大小 |
| `--output-dir` | str | `./eval_results` | 输出目录 |
| `--format` | str | `json` | 输出格式（json/markdown/csv） |
| `--limit` | int | None | 限制评估样本数（调试用） |
| `--seed` | int | 42 | 随机种子 |

**支持的评估基准：**

| 基准 | 说明 | 指标 |
|------|------|------|
| `mmlu` | 大规模多任务语言理解 | 准确率 |
| `ceval` | 中文综合评估 | 准确率 |
| `humaneval` | 代码生成 | Pass@1, Pass@10 |
| `gsm8k` | 数学推理 | 准确率 |
| `hellaswag` | 语言推理 | 准确率 |
| `arc` | AI2 推理挑战 | 准确率 |
| `truthfulqa` | 真实性问答 | 准确率 |
| `winogrande` | 常识推理 | 准确率 |

---

## 常见问题

### Q: 安装时 CUDA 相关错误

```bash
# 确认 CUDA 版本
nvcc --version

# 安装匹配的 PyTorch 版本
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### Q: 训练时显存不足 (OOM)

```bash
# 方案 1：减小批次大小，增加梯度累积
nexus-train --batch-size 1 --gradient-accumulation-steps 16 ...

# 方案 2：启用梯度检查点
nexus-train --gradient-checkpointing ...

# 方案 3：使用 LoRA 微调
nexus-train --lora-r 16 --lora-alpha 32 ...

# 方案 4：使用 DeepSpeed ZeRO-3
nexus-train --strategy deepspeed_stage3 --num-gpus 4 ...
```

### Q: 推理服务启动失败

```bash
# 检查 GPU 可用性
python -c "import torch; print(torch.cuda.is_available())"

# 检查端口是否被占用
lsof -i :8000

# 查看详细日志
nexus-serve --model-path nexus-ai/nexus-7b --log-level debug
```

### Q: 如何切换到 BF16 训练

```bash
# BF16 在 A100/H100 等 Ampere 及以上架构 GPU 上推荐使用
nexus-train --bf16 --batch-size 8 ...
```

---

> 更多详细信息请参阅 [API 参考](./api_reference.md)、[架构设计](./architecture.md) 和 [部署指南](./deployment.md)。
