# Nexus-LLM 架构设计文档

> 本文档详细描述 Nexus-LLM 框架的整体架构、模块依赖关系、核心模型设计以及训练/推理流水线。

---

## 目录

- [整体架构](#整体架构)
- [模块依赖关系](#模块依赖关系)
- [模型架构详解](#模型架构详解)
  - [Transformer 层](#transformer-层)
  - [分组查询注意力 (GQA)](#分组查询注意力-gqa)
  - [旋转位置编码 (RoPE)](#旋转位置编码-rope)
  - [SwiGLU 激活函数](#swiglu-激活函数)
  - [RMSNorm 归一化](#rmsnorm-归一化)
- [训练流水线架构](#训练流水线架构)
- [推理服务架构](#推理服务架构)
- [数据流图](#数据流图)

---

## 整体架构

Nexus-LLM 采用分层模块化架构，从底层到顶层分为五个核心层级：

```
+=====================================================================+
|                        Nexus-LLM 整体架构                            |
+=====================================================================+
|                                                                     |
|  +-----------------------------+  +-------------------------------+ |
|  |       应用层 (Application)   |  |     工具层 (Tooling)          | |
|  |                             |  |                               | |
|  |  - nexus-train (CLI)        |  |  - nexus-eval (评估)          | |
|  |  - nexus-serve (服务)       |  |  - Agent 框架                 | |
|  |  - Python API               |  |  - 压缩工具                   | |
|  +-------------+---------------+  +---------------+---------------+ |
|                |                                  |                 |
|  +-------------+---------------+  +---------------+---------------+ |
|  |       服务层 (Service)       |  |     增强层 (Enhancement)      | |
|  |                             |  |                               | |
|  |  - InferenceEngine          |  |  - RLHF (PPO/DPO)            | |
|  |  - ModelServer              |  |  - 增量学习 (EWC/Adapter)     | |
|  |  - OpenAI Compatible API    |  |  - 模型压缩 (蒸馏/剪枝/量化)  | |
|  +-------------+---------------+  +---------------+---------------+ |
|                |                                  |                 |
|  +-------------+----------------------------------+---------------+ |
|  |                    训练层 (Training)                           | |
|  |                                                             | |
|  |  - Trainer / DistributedTrainer / FineTuningTrainer         | |
|  |  - DeepSpeed / FSDP 集成                                     | |
|  |  - 数据加载与预处理                                          | |
|  |  - 学习率调度 / 梯度裁剪 / 混合精度                          | |
|  +--------------------------------------+----------------------+ |
|                |                          |                     |
|  +-------------+--------------------------+---------------------+ |
|  |                    核心层 (Core)                              | |
|  |                                                             | |
|  |  - NexusConfig (模型配置)                                   | |
|  |  - NexusModel (基础 Transformer)                            | |
|  |  - NexusForCausalLM (因果语言模型)                          | |
|  |  - NexusTokenizer (分词器)                                  | |
|  |  - 注意力机制 (GQA) / 位置编码 (RoPE)                       | |
|  |  - 激活函数 (SwiGLU) / 归一化 (RMSNorm)                     | |
|  +-------------------------------------------------------------+ |
|                                                                     |
+=====================================================================+
```

### 设计原则

| 原则 | 说明 |
|------|------|
| **模块化** | 各层之间通过清晰的接口通信，可独立替换和升级 |
| **可扩展** | 通过注册机制支持自定义组件（工具、回调、损失函数等） |
| **高性能** | 内核层使用 CUDA Kernel 优化，支持 Flash Attention 和 PagedAttention |
| **易用性** | 提供高层 Python API 和 CLI 工具，降低使用门槛 |
| **兼容性** | 与 HuggingFace 生态无缝集成，支持 OpenAI API 格式 |

---

## 模块依赖关系

```
nexus_llm/
|
|-- __init__.py                    # 顶层导出
|
|-- core/                          # 核心模型模块
|   |-- __init__.py
|   |-- config.py                  # NexusConfig
|   |-- model.py                   # NexusModel, NexusForCausalLM
|   |-- tokenizer.py               # NexusTokenizer
|   |-- attention.py               # GQA 注意力机制
|   |-- embedding.py               # 词嵌入 + 位置编码
|   |-- mlp.py                     # SwiGLU FFN
|   |-- normalization.py           # RMSNorm
|   |-- parallel.py                # 张量/流水线并行
|   `-- distributed/               # 分布式通信原语
|       |-- tensor_parallel.py
|       `-- pipeline_parallel.py
|
|-- training/                      # 训练模块
|   |-- __init__.py
|   |-- trainer.py                 # Trainer 基类
|   |-- distributed_trainer.py     # DistributedTrainer
|   |-- finetuning_trainer.py      # FineTuningTrainer (LoRA/QLoRA)
|   |-- config.py                  # TrainingConfig
|   |-- data.py                    # 数据加载与预处理
|   |-- scheduler.py               # 学习率调度器
|   |-- callbacks.py               # 训练回调
|   `-- optimizers/                # 优化器
|       |-- adamw.py
|       `-- schedule_free.py
|
|-- serving/                       # 推理服务模块
|   |-- __init__.py
|   |-- engine.py                  # InferenceEngine
|   |-- server.py                  # ModelServer (FastAPI)
|   |-- openai_api.py              # OpenAI 兼容 API 路由
|   |-- batching.py                # 连续批处理
|   |-- cache.py                   # PagedAttention KV Cache
|   `-- metrics.py                 # 服务指标
|
|-- rlhf/                          # RLHF 模块
|   |-- __init__.py
|   |-- reward_model.py            # RewardModel
|   |-- ppo_trainer.py             # PPOTrainer
|   |-- dpo_trainer.py             # DPOTrainer
|   |-- config.py                  # RLHFConfig, PPOConfig
|   `-- dataset.py                 # 偏好数据集处理
|
|-- agent/                         # Agent 模块
|   |-- __init__.py
|   |-- react_agent.py             # ReActAgent
|   |-- function_calling_agent.py  # FunctionCallingAgent
|   |-- tool_registry.py           # ToolRegistry
|   |-- base_tool.py               # BaseTool
|   |-- multi_agent.py             # 多 Agent 协调
|   `-- tools/                     # 内置工具
|       |-- search.py
|       |-- calculator.py
|       |-- code_executor.py
|       |-- file_reader.py
|       `-- web_browser.py
|
|-- compression/                   # 模型压缩模块
|   |-- __init__.py
|   |-- distiller.py               # KnowledgeDistiller
|   |-- pruner.py                  # StructuredPruner
|   |-- quantizer.py               # Quantizer (GPTQ/AWQ/FP8)
|   |-- compressor.py              # ModelCompressor
|   `-- config.py
|
|-- incremental/                   # 增量学习模块
|   |-- __init__.py
|   |-- incremental_trainer.py     # IncrementalTrainer
|   |-- regularizers.py            # EWC/SI/MAS 正则化器
|   |-- adapters.py                # AdapterModule
|   |-- replay_buffer.py           # 经验回放缓冲区
|   `-- config.py
|
|-- evaluation/                    # 评估模块
|   |-- __init__.py
|   |-- evaluator.py               # Evaluator
|   |-- benchmarks/                # 基准测试
|   |   |-- mmlu.py
|   |   |-- ceval.py
|   |   |-- humaneval.py
|   |   `-- ...
|   `-- metrics.py
|
`-- cli/                           # 命令行工具
    |-- train.py                   # nexus-train
    |-- serve.py                   # nexus-serve
    `-- eval.py                    # nexus-eval
```

### 模块依赖图

```
                    +-----------+
                    |   CLI     |
                    | (nexus-*) |
                    +-----+-----+
                          |
              +-----------+-----------+
              |                       |
        +-----+-----+          +-----+-----+
        | Training  |          | Serving   |
        |  Module   |          |  Module   |
        +-----+-----+          +-----+-----+
              |                       |
              |              +--------+--------+
              |              |                 |
        +-----+-----+  +----+----+    +-------+------+
        | RLHF      |  | Agent   |    | Compression  |
        | Module    |  | Module  |    | Module       |
        +-----+-----+  +----+----+    +-------+------+
              |              |                 |
              |       +------+------+         |
              |       | Incremental |         |
              |       | Learning    |         |
              |       +------+------+         |
              |              |                 |
        +-----+--------------+-----------------+-----+
        |              Core Module                    |
        |  (NexusConfig, NexusModel, Attention, ...)  |
        +---------------------------------------------+
                          |
              +-----------+-----------+
              |      PyTorch         |
              |  (CUDA, Distributed) |
              +-----------------------+
```

---

## 模型架构详解

### Transformer 层

Nexus-LLM 基于 Decoder-only Transformer 架构，每个 Transformer 层包含以下子模块：

```
输入: hidden_states (batch_size, seq_len, hidden_size)
                    |
                    v
          +-------------------+
          | RMSNorm (Pre-Norm) |
          +-------------------+
                    |
                    v
          +-------------------+
          |   Self-Attention  |
          |   (GQA + RoPE)    |
          +-------------------+
                    |
                    v
              +---------+
              | Dropout |
              +---------+
                    |
                    v
         hidden_states + attention_output
                    |
                    v
          +-------------------+
          | RMSNorm (Pre-Norm) |
          +-------------------+
                    |
                    v
          +-------------------+
          |      SwiGLU       |
          |   (FFN 层)        |
          +-------------------+
                    |
                    v
              +---------+
              | Dropout |
              +---------+
                    |
                    v
         hidden_states + ffn_output
                    |
                    v
输出: hidden_states (batch_size, seq_len, hidden_size)
```

**代码结构：**

```python
class NexusDecoderLayer(nn.Module):
    def __init__(self, config: NexusConfig):
        self.input_layernorm = NexusRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = NexusGQAAttention(config)
        self.post_attention_layernorm = NexusRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = NexusSwiGLUMLP(config)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        # Pre-Norm + Self-Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, _, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + self.dropout(attn_output)

        # Pre-Norm + FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_output = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(ffn_output)

        return hidden_states, present_key_value
```

---

### 分组查询注意力 (GQA)

GQA (Grouped-Query Attention) 是多头注意力 (MHA) 和多查询注意力 (MQA) 的折中方案。它将 Query 头分为若干组，每组共享一组 Key/Value 头。

```
标准 MHA (num_heads = 32, kv_heads = 32):
+--------+--------+--------+--------+  (32 组独立的 Q, K, V)
| Q1 K1 V1 | Q2 K2 V2 | Q3 K3 V3 | ... |
+--------+--------+--------+--------+

GQA (num_heads = 32, kv_heads = 8):
+--------+--------+--------+--------+  (32 个 Q 头，8 组共享的 K, V)
| Q1     | Q2     | Q3     | Q4     |
|  +--+  |  +--+  |  +--+  |  +--+  |
|  |K1V1| |  |K2V2| |  |K3V3| |  |K4V4| |
|  +--+  |  +--+  |  +--+  |  +--+  |
| Q5     | Q6     | Q7     | Q8     |
|  +--+  |  +--+  |  +--+  |  +--+  |
|  |K5V5| |  |K6V6| |  |K7V7| |  |K8V8| |
|  +--+  |  +--+  |  +--+  |  +--+  |
+--------+--------+--------+--------+

MQA (num_heads = 32, kv_heads = 1):
+--------+--------+--------+--------+  (32 个 Q 头，1 组共享的 K, V)
| Q1     | Q2     | Q3     | ...    |
|  +--+  |  +--+  |  +--+  |        |
|  |K V| |  |K V| |  |K V| |        |
|  +--+  |  +--+  |  +--+  |        |
+--------+--------+--------+--------+
```

**优势对比：**

| 方案 | KV Cache 大小 | 推理速度 | 模型质量 |
|------|-------------|---------|---------|
| MHA | 大 (1x) | 慢 | 最高 |
| GQA | 中 (1/4) | 快 | 接近 MHA |
| MQA | 小 (1/32) | 最快 | 略低 |

**计算过程：**

```python
class NexusGQAAttention(nn.Module):
    def __init__(self, config: NexusConfig):
        self.num_heads = config.num_attention_heads          # 32
        self.num_kv_heads = config.num_key_value_heads       # 8
        self.head_dim = config.hidden_size // self.num_heads # 128
        self.num_groups = self.num_heads // self.num_kv_heads # 4

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size)

    def forward(self, hidden_states, attention_mask, position_ids, past_key_value):
        batch_size, seq_len, _ = hidden_states.shape

        # 投影: Q -> (B, T, 32, 128), K/V -> (B, T, 8, 128)
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 应用 RoPE 位置编码
        q, k = apply_rotary_pos_emb(q, k, position_ids)

        # 扩展 KV 头以匹配 Q 头数: (B, T, 8, 128) -> (B, T, 32, 128)
        k = k.repeat_interleave(self.num_groups, dim=2)
        v = v.repeat_interleave(self.num_groups, dim=2)

        # 计算 Flash Attention
        attn_output = flash_attention(q, k, v, attention_mask)

        # 输出投影
        output = self.o_proj(attn_output)
        return output
```

---

### 旋转位置编码 (RoPE)

RoPE (Rotary Position Embedding) 通过旋转矩阵对 Query 和 Key 施加位置信息，具有相对位置编码的优良特性。

**数学原理：**

对于位置 `m` 和维度对 `(2i, 2i+1)`，RoPE 定义旋转角度为：

```
theta_i = m * theta^(-2i/d)

其中 theta = 10000.0（基础频率），d 为 head_dim
```

旋转操作：

```
q'[2i]   = q[2i] * cos(theta_i) - q[2i+1] * sin(theta_i)
q'[2i+1] = q[2i] * sin(theta_i) + q[2i+1] * cos(theta_i)
```

**可视化：**

```
维度 0,1 (theta_0 = m * 10000^(-0/128)):
    q'[0] = q[0] * cos(theta_0) - q[1] * sin(theta_0)
    q'[1] = q[0] * sin(theta_0) + q[1] * cos(theta_0)

维度 2,3 (theta_1 = m * 10000^(-2/128)):
    q'[2] = q[2] * cos(theta_1) - q[3] * sin(theta_1)
    q'[3] = q[2] * sin(theta_1) + q[3] * cos(theta_1)

    ...

维度 126,127 (theta_63 = m * 10000^(-126/128)):
    q'[126] = q[126] * cos(theta_63) - q[127] * sin(theta_63)
    q'[127] = q[126] * sin(theta_63) + q[127] * cos(theta_63)
```

**实现：**

```python
def build_rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0,
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos 和 sin 缓存。"""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim/2)
    cos_cache = freqs.cos()        # (seq_len, head_dim/2)
    sin_cache = freqs.sin()        # (seq_len, head_dim/2)
    return cos_cache, sin_cache


def apply_rotary_pos_emb(q, k, cos, sin):
    """对 Q 和 K 应用旋转位置编码。"""
    # q, k: (batch, num_heads, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim/2)
    q_reshape = q.float().reshape(*q.shape[:-1], -1, 2)  # (..., head_dim/2, 2)
    k_reshape = k.float().reshape(*k.shape[:-1], -1, 2)

    # 旋转
    q_out = torch.stack([
        q_reshape[..., 0] * cos - q_reshape[..., 1] * sin,
        q_reshape[..., 0] * sin + q_reshape[..., 1] * cos,
    ], dim=-1).flatten(-2)

    k_out = torch.stack([
        k_reshape[..., 0] * cos - k_reshape[..., 1] * sin,
        k_reshape[..., 0] * sin + k_reshape[..., 1] * cos,
    ], dim=-1).flatten(-2)

    return q_out.type_as(q), k_out.type_as(k)
```

**RoPE 的关键特性：**

| 特性 | 说明 |
|------|------|
| 相对位置 | 注意力分数仅依赖 token 间的相对距离 |
| 长度外推 | 支持通过 NTK-Aware 或 YaRN 方法扩展上下文长度 |
| 计算高效 | 可预计算 cos/sin 缓存，推理时零额外开销 |
| 无需学习 | 位置信息完全由旋转角度决定，无需学习参数 |

---

### SwiGLU 激活函数

SwiGLU 是 GLU (Gated Linear Unit) 的变体，使用 Swish 作为门控激活函数，在 LLaMA 等模型中被广泛采用。

**数学定义：**

```
SwiGLU(x, W, V, b, c, W2) = (Swish(xW + b) * (xV + c)) W2

其中:
  Swish(x) = x * sigmoid(x) = x / (1 + e^(-x))
```

**结构图：**

```
输入 x (hidden_size)
        |
        +------------------+
        |                  |
        v                  v
  +-----------+    +-----------+
  | W_gate    |    | W_up      |
  | (d -> d_ff)|    | (d -> d_ff)|
  +-----------+    +-----------+
        |                  |
        v                  v
  +-----------+            |
  | Swish     |            |
  | (x*sig(x))|            |
  +-----------+            |
        |                  |
        v                  v
      (element-wise multiply)
        |
        v
  +-----------+
  | W_down    |
  | (d_ff->d) |
  +-----------+
        |
        v
输出 (hidden_size)
```

**实现：**

```python
class NexusSwiGLUMLP(nn.Module):
    def __init__(self, config: NexusConfig):
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()  # Swish 激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate_proj -> Swish, up_proj -> 逐元素相乘, down_proj -> 输出
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**与其他激活函数对比：**

| 激活函数 | 公式 | 参数量 | 性能 |
|---------|------|--------|------|
| ReLU | max(0, x) | 3d | 基准 |
| GELU | x * Phi(x) | 3d | 较好 |
| GLU | (xW1) * sigmoid(xW2) | 4d | 好 |
| SwiGLU | Swish(xW1) * (xW2) | 4d | 最好 |

> SwiGLU 虽然增加了约 33% 的 FFN 参数量，但在相同模型规模下能获得更好的性能。

---

### RMSNorm 归一化

RMSNorm 是 LayerNorm 的高效替代方案，仅使用均方根进行归一化，省去了均值计算和偏置参数。

**数学定义：**

```
RMSNorm(x) = x / RMS(x) * gamma

其中:
  RMS(x) = sqrt(mean(x^2) + eps)
  gamma: 可学习的缩放参数
```

**与 LayerNorm 对比：**

```
LayerNorm:
  x_hat = (x - mean(x)) / sqrt(var(x) + eps)
  output = gamma * x_hat + beta
  参数: gamma (d), beta (d) -> 2d 参数

RMSNorm:
  x_hat = x / sqrt(mean(x^2) + eps)
  output = gamma * x_hat
  参数: gamma (d) -> d 参数
```

**实现：**

```python
class NexusRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, hidden_size)
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).type_as(x)
```

**性能优势：**

| 指标 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 计算量 | 高（需计算均值和方差） | 低（仅需均方根） |
| 参数量 | 2d | d |
| 训练速度 | 基准 | 快 7-10% |
| 模型质量 | 基准 | 相当 |

---

## 训练流水线架构

```
+=====================================================================+
|                        训练流水线架构                                |
+=====================================================================+
                                                                     |
  +------------------+     +-------------------+     +--------------+ |
  |   数据准备阶段    |     |   训练执行阶段     |     |  保存与评估   | |
  |                  |     |                   |     |              | |
  |  +------------+  |     |  +-------------+  |     | +----------+ | |
  |  | JSONL/CSV  |  |     |  | 前向传播    |  |     | | 模型保存 | | |
  |  | 数据文件   |->+--+  |  | (FP16/BF16) |  |     | |          | | |
  |  +------------+  |  |  |  +------+------+  |     | +----+-----+ | |
  |                  |  |  |       |           |     |      |       | |
  |  +------------+  |  |  |  +----v------+    |     | +----v-----+ | |
  |  | 数据预处理  |  |  |  |  | 损失计算   |    |     | | 评估运行 | | |
  |  | - Tokenize |  |  |  |  | (CrossEntropy)|   |     | |          | | |
  |  | - Padding   |->+--+  |  +------+------+    |     | +----+-----+ | |
  |  | - Truncation|     |  |       |           |     |      |       | |
  |  +------------+     |  |  +----v------+    |     | +----v-----+ | |
  |                     |  |  | 反向传播    |    |     | | 指标记录 | | |
  |                     |  |  | (梯度计算)  |    |     | |          | | |
  |                     |  |  +------+------+    |     | +----+-----+ | |
  |                     |  |       |           |     |      |       | |
  |                     |  |  +----v------+    |     | +----v-----+ | |
  |                     |  |  | 梯度累积   |    |     | | 回调触发 | | |
  |                     |  |  | (N 步)     |    |     | |          | | |
  |                     |  |  +------+------+    |     | +----------+ | |
  |                     |  |       |           |     |              | |
  |                     |  |  +----v------+    |     |              | |
  |                     |  |  | 梯度裁剪   |    |     |              | |
  |                     |  |  | (max_norm) |    |     |              | |
  |                     |  |  +------+------+    |     |              | |
  |                     |  |       |           |     |              | |
  |                     |  |  +----v------+    |     |              | |
  |                     |  |  | 优化器更新 |    |     |              | |
  |                     |  |  | (AdamW)   |    |     |              | |
  |                     |  |  +------------+    |     |              | |
  |                     |  +-------------------+     +--------------+ |
  +------------------+                                            |
                                                                     |
  +---------------------------------------------------------------+ |
  |                    分布式训练策略                                | |
  |                                                                | |
  |  +------------------+  +------------------+  +---------------+  | |
  |  | DDP (数据并行)   |  | DeepSpeed ZeRO   |  | FSDP          |  | |
  |  |                  |  |                  |  |               |  | |
  |  | - 每GPU完整模型  |  | Stage1: 优化器分片|  | - 按层分片    |  | |
  |  | - 梯度AllReduce  |  | Stage2: +梯度分片 |  | - 自动收集    |  | |
  |  | - 简单易用       |  | Stage3: +模型分片|  | - 内存高效    |  | |
  |  +------------------+  +------------------+  +---------------+  | |
  +---------------------------------------------------------------+ |
                                                                     |
+=====================================================================+
```

### 训练数据流

```
原始数据 (JSONL)
    |
    v
+-------------------+
| 数据加载器         |  dataloader_num_workers=4
| (多进程预取)       |  pin_memory=True
+-------------------+
    |
    v
+-------------------+
| Tokenizer 编码     |  add_special_tokens=True
| (批量处理)        |  max_length=2048
+-------------------+
    |
    v
+-------------------+
| 数据整理器         |  padding="max_length"
| (DataCollator)    |  return_tensors="pt"
+-------------------+
    |
    v
+-------------------+
| 梯度累积缓冲区     |  gradient_accumulation_steps=4
| (模拟大 batch)    |  effective_batch = 4 * 4 = 16
+-------------------+
    |
    v
+-------------------+
| 模型前向传播       |  fp16=True / bf16=True
| (混合精度)        |  gradient_checkpointing=True
+-------------------+
    |
    v
+-------------------+
| 损失计算           |  label_smoothing=0.0
| (CrossEntropy)    |  ignore_index=-100
+-------------------+
    |
    v
+-------------------+
| 反向传播 + 裁剪    |  max_grad_norm=1.0
+-------------------+
    |
    v
+-------------------+
| 优化器更新         |  AdamW (lr=2e-5, weight_decay=0.01)
| (学习率调度)       |  CosineAnnealing + Warmup
+-------------------+
```

---

## 推理服务架构

```
+=====================================================================+
|                        推理服务架构                                  |
+=====================================================================+
                                                                     |
  客户端请求                                                          |
  (OpenAI SDK / curl / Python)                                        |
        |                                                            |
        v                                                            |
  +------------------------------------------------------------------+ |
  |                     ModelServer (FastAPI)                         | |
  |                                                                  | |
  |  +------------------+  +------------------+  +-----------------+ | |
  |  | 认证中间件        |  | 限流中间件        | | 日志中间件       | | |
  |  | (API Key)        |  | (Token Bucket)   | | (Request Log)   | | |
  |  +------------------+  +------------------+  +-----------------+ | |
  |                                                                  | |
  |  +-----------------------------------------------------------+  | |
  |  |                    API 路由层                                |  | |
  |  |                                                           |  | |
  |  |  POST /v1/chat/completions  -----> ChatCompletionHandler  |  | |
  |  |  POST /v1/completions        -----> CompletionHandler     |  | |
  |  |  GET  /v1/models             -----> ModelListHandler      |  | |
  |  |  POST /v1/embeddings         -----> EmbeddingHandler      |  | |
  |  |  GET  /health                -----> HealthCheckHandler    |  | |
  |  +-----------------------------------------------------------+  | |
  |                                                                  | |
  |  +-----------------------------------------------------------+  | |
  |  |                    InferenceEngine                          |  | |
  |  |                                                           |  | |
  |  |  +------------------+  +------------------+               |  | |
  |  |  | 请求队列         |  | 连续批处理器       |               |  | |
  |  |  | (Priority Queue) |  | (Continuous       |               |  | |
  |  |  |                  |  |  Batching)        |               |  | |
  |  |  +--------+---------+  +--------+---------+               |  | |
  |  |           |                     |                          |  | |
  |  |           v                     v                          |  | |
  |  |  +------------------------------------------+             |  | |
  |  |  |         PagedAttention KV Cache           |             |  | |
  |  |  |  +------+ +------+ +------+ +------+     |             |  | |
  |  |  |  |Page 0| |Page 1| |Page 2| |Page 3| ... |             |  | |
  |  |  |  |4KB   | |4KB   | |4KB   | |4KB   |     |             |  | |
  |  |  |  +------+ +------+ +------+ +------+     |             |  | |
  |  |  |  虚拟地址 -> 物理页映射 (页表管理)        |             |  | |
  |  |  +------------------------------------------+             |  | |
  |  |                        |                               |  | |
  |  |                        v                               |  | |
  |  |  +------------------------------------------+             |  | |
  |  |  |         GPU 推理引擎                      |             |  | |
  |  |  |  - CUDA Kernel 优化                       |             |  | |
  |  |  |  - Flash Attention 2                     |             |  | |
  |  |  |  - Tensor Parallel (多 GPU)              |             |  | |
  |  |  |  - 量化推理 (AWQ/GPTQ/FP8)              |             |  | |
  |  |  +------------------------------------------+             |  | |
  |  +-----------------------------------------------------------+  | |
  |                                                                  | |
  |  +------------------+  +------------------+                      | |
  |  | Prometheus 指标   |  | 健康检查          |                      | |
  |  | (/metrics)        |  | (/health)         |                      | |
  |  +------------------+  +------------------+                      | |
  +------------------------------------------------------------------+
                                                                     |
+=====================================================================+
```

### 连续批处理 (Continuous Batching)

传统静态批处理中，需要等待整个批次的所有请求完成后才能处理下一批。连续批处理允许在请求完成后立即插入新请求，大幅提升吞吐量。

```
时间轴 ->

静态批处理:
  Batch 1: [Req A (5 tokens)] [Req B (3 tokens)] [Req C (8 tokens)]
           |<=================等待最长的请求=================>|
  Batch 2: [Req D (4 tokens)] [Req E (6 tokens)]
           |<=================等待最长的请求=================>|

连续批处理:
  Step 1: [Req A] [Req B] [Req C]
  Step 2: [Req A] [Req B] [Req C]
  Step 3: [Req A] [    ] [Req C] [Req D]  <- B 完成，D 加入
  Step 4: [Req A] [    ] [Req C] [Req D]
  Step 5: [    ] [    ] [Req C] [Req D] [Req E]  <- A 完成，E 加入
  Step 6: [    ] [    ] [Req C] [Req D] [Req E]
  ...
```

### PagedAttention KV Cache

PagedAttention 将 KV Cache 划分为固定大小的页（类似操作系统的虚拟内存），避免内存碎片和预分配浪费。

```
传统 KV Cache:
  Request 1: [KV Block 1][KV Block 2][KV Block 3][  空闲  ][  空闲  ]
  Request 2: [KV Block 1][KV Block 2][  空闲  ][  空闲  ][  空闲  ]
  -> 每个请求预分配最大长度，造成大量浪费

PagedAttention:
  物理内存: [Page 0][Page 1][Page 2][Page 3][Page 4][Page 5][Page 6]

  Request 1 虚拟地址: [vPage 0][vPage 1][vPage 2]
                       |          |          |
  页表映射:             v          v          v
                       [Page 0]  [Page 2]  [Page 5]

  Request 2 虚拟地址: [vPage 0][vPage 1]
                       |          |
  页表映射:             v          v
                       [Page 1]  [Page 3]

  -> 按需分配，无浪费，支持动态序列长度
```

---

## 数据流图

### 端到端训练数据流

```
+----------+     +----------+     +----------+     +----------+
|  原始数据  |---->|  清洗    |---->|  分词    |---->|  格式化   |
| (JSONL)  |     | (Filter) |     | (Token)  |     | (Tensor) |
+----------+     +----------+     +----------+     +----------+
                                                       |
                                                       v
+----------+     +----------+     +----------+     +----------+
|  保存     |<----|  评估    |<----|  优化    |<----|  前向    |
| (Save)   |     | (Eval)   |     | (Update) |     | (Forward)|
+----------+     +----------+     +----------+     +----------+
                    |                |
                    v                v
              +----------+     +----------+
              |  日志     |     |  反向    |
              | (Logging)|     | (Backward)|
              +----------+     +----------+
```

### 端到端推理数据流

```
+----------+     +----------+     +----------+     +----------+
| HTTP 请求  |---->|  认证    |---->|  解析    |---->|  分词    |
| (JSON)   |     | (Auth)   |     | (Parse)  |     | (Token)  |
+----------+     +----------+     +----------+     +----------+
                                                       |
                                                       v
+----------+     +----------+     +----------+     +----------+
| HTTP 响应  |<---->|  反分词  |<---->|  采样    |<---->|  模型    |
| (JSON)   |     | (Decode) |     | (Sample) |     | (Model)  |
+----------+     +----------+     +----------+     +----------+
                                                       |
                                                       v
                                                 +----------+
                                                 | KV Cache |
                                                 | (Paged)  |
                                                 +----------+
```

### RLHF 训练数据流

```
+----------+     +----------+     +----------+     +----------+
|  提示数据  |---->|  策略模型 |---->|  生成    |---->|  奖励模型 |
| (Prompt) |     | (Policy) |     | (Generate)|     | (Reward)  |
+----------+     +----------+     +----------+     +----------+
                                                       |
                                                       v
+----------+     +----------+     +----------+     +----------+
|  更新     |<---->|  PPO     |<---->|  价值模型 |<----|  参考模型 |
| (Update) |     | (Compute)|     | (Value)   |     | (Ref)     |
+----------+     +----------+     +----------+     +----------+
```

---

> 更多信息请参阅 [API 参考](./api_reference.md) 和 [部署指南](./deployment.md)。
