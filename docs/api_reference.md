# Nexus-LLM API 参考文档

> 本文档详细介绍 Nexus-LLM 框架的所有公开 API 接口，包括核心模型、训练、推理服务、RLHF、Agent 和模型压缩等模块。

---

## 目录

- [核心模型 API](#核心模型-api)
  - [NexusConfig](#nexusconfig)
  - [NexusModel](#nexusmodel)
  - [NexusForCausalLM](#nexusforcausallm)
  - [NexusTokenizer](#nexustokenizer)
- [训练 API](#训练-api)
  - [TrainingConfig](#trainingconfig)
  - [Trainer](#trainer)
  - [DistributedTrainer](#distributedtrainer)
  - [FineTuningTrainer](#finetuningtrainer)
- [推理服务 API](#推理服务-api)
  - [InferenceEngine](#inferenceengine)
  - [ModelServer](#modelserver)
  - [OpenAI 兼容 API](#openai-兼容-api)
- [RLHF API](#rlhf-api)
  - [RewardModel](#rewardmodel)
  - [PPOTrainer](#ppotrainer)
  - [RLHFConfig](#rlhfconfig)
- [Agent API](#agent-api)
  - [ReActAgent](#reactagent)
  - [FunctionCallingAgent](#functioncallingagent)
  - [ToolRegistry](#toolregistry)
  - [BaseTool](#basetool)
- [压缩 API](#压缩-api)
  - [KnowledgeDistiller](#knowledgedistiller)
  - [StructuredPruner](#structuredpruner)
  - [ModelCompressor](#modelcompressor)
- [增量学习 API](#增量学习-api)
  - [IncrementalTrainer](#incrementaltrainer)
  - [EWCRegularizer](#ewcregularizer)
  - [AdapterModule](#adaptermodule)

---

## 核心模型 API

### NexusConfig

模型配置类，定义模型的所有超参数。

```python
from nexus_llm import NexusConfig
```

**类定义：**

```python
class NexusConfig:
    """Nexus-LLM 模型配置。

    参数:
        model_name (str): 模型名称标识。
        vocab_size (int): 词表大小。默认: 32000
        hidden_size (int): 隐藏层维度。默认: 4096
        intermediate_size (int): FFN 中间层维度。默认: 11008
        num_hidden_layers (int): Transformer 层数。默认: 32
        num_attention_heads (int): 注意力头数。默认: 32
        num_key_value_heads (int): GQA 键值头数。默认: None（等同于 num_attention_heads）
        max_position_embeddings (int): 最大位置编码长度。默认: 4096
        rms_norm_eps (float): RMSNorm epsilon。默认: 1e-6
        rope_theta (float): RoPE 基础频率。默认: 10000.0
        hidden_act (str): 隐藏层激活函数。默认: "silu"
        tie_word_embeddings (bool): 是否共享输入输出词嵌入。默认: False
        attention_dropout (float): 注意力 Dropout。默认: 0.0
        initializer_range (float): 参数初始化范围。默认: 0.02
        use_cache (bool): 是否启用 KV Cache。默认: True
        bos_token_id (int): BOS token ID。默认: 1
        eos_token_id (int): EOS token ID。默认: 2
        pad_token_id (int): PAD token ID。默认: 0
    """
```

**使用示例：**

```python
# 从预设创建配置
config = NexusConfig.from_pretrained("nexus-ai/nexus-7b")

# 手动创建配置
config = NexusConfig(
    model_name="nexus-7b",
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,        # 启用 GQA，KV 头数为 8
    num_hidden_layers=32,
    intermediate_size=11008,
    max_position_embeddings=8192,
)

# 修改配置
config.hidden_size = 5120
config.num_hidden_layers = 48

# 保存配置
config.save_pretrained("./my-config")

# 从 JSON 文件加载
config = NexusConfig.from_json_file("./config.json")
```

---

### NexusModel

Nexus-LLM 的基础 Transformer 模型（不含 LM Head）。

```python
from nexus_llm import NexusModel
```

**类定义：**

```python
class NexusModel(nn.Module):
    """Nexus-LLM 基础模型。

    包含嵌入层、多层 Transformer 编码器/解码器和最终的 LayerNorm。
    不包含语言模型头（LM Head）。

    参数:
        config (NexusConfig): 模型配置实例。
    """

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        """前向传播。

        参数:
            input_ids: 输入 token ID，形状为 (batch_size, seq_length)。
            attention_mask: 注意力掩码，形状为 (batch_size, seq_length)。
            position_ids: 位置 ID，形状为 (batch_size, seq_length)。
            past_key_values: 过去的 KV Cache，用于增量生成。
            inputs_embeds: 直接传入的嵌入表示（与 input_ids 二选一）。
            use_cache: 是否返回 KV Cache。

        返回:
            BaseModelOutputWithPast:
                - last_hidden_state: (batch_size, seq_length, hidden_size)
                - past_key_values: 更新后的 KV Cache
                - hidden_states: 所有层的隐藏状态（可选）
                - attentions: 所有层的注意力权重（可选）
        """
```

**使用示例：**

```python
model = NexusModel.from_pretrained("nexus-ai/nexus-7b")

# 获取隐藏状态
outputs = model(
    input_ids=tokenizer("你好", return_tensors="pt").input_ids,
    use_cache=False,
)
hidden_states = outputs.last_hidden_state  # (1, seq_len, 4096)
```

---

### NexusForCausalLM

用于因果语言建模的完整模型（包含 LM Head）。

```python
from nexus_llm import NexusForCausalLM
```

**类定义：**

```python
class NexusForCausalLM(nn.Module):
    """Nexus-LLM 因果语言模型。

    在 NexusModel 基础上增加了线性 LM Head，用于下一个 token 预测。
    支持从预训练权重加载、梯度检查点和 KV Cache。

    参数:
        config (NexusConfig): 模型配置实例。
    """

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """前向传播。

        参数:
            input_ids: 输入 token ID。
            attention_mask: 注意力掩码。
            position_ids: 位置 ID。
            past_key_values: KV Cache。
            labels: 标签 token ID（用于计算损失）。
            use_cache: 是否返回 KV Cache。

        返回:
            CausalLMOutputWithPast:
                - loss: 语言模型损失（当提供 labels 时）
                - logits: (batch_size, seq_length, vocab_size)
                - past_key_values: 更新后的 KV Cache
        """

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """自回归生成。

        参数:
            input_ids: 输入 token ID。
            max_new_tokens: 最大生成 token 数。
            temperature: 采样温度。
            top_p: nucleus 采样概率阈值。
            top_k: top-k 采样参数。
            repetition_penalty: 重复惩罚。
            do_sample: 是否采样（False 则使用贪心解码）。
            eos_token_id: 结束 token ID。
            pad_token_id: 填充 token ID。

        返回:
            torch.LongTensor: 生成的 token ID 序列。
        """

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "NexusForCausalLM":
        """从预训练权重加载模型。

        参数:
            model_path: 本地路径或 HuggingFace 模型 ID。
            **kwargs: 额外参数（如 torch_dtype, device_map 等）。

        返回:
            NexusForCausalLM: 加载完成的模型实例。
        """

    def save_pretrained(self, output_dir: str, safe_serialization: bool = True):
        """保存模型权重和配置。

        参数:
            output_dir: 保存目录。
            safe_serialization: 是否使用 safetensors 格式保存。
        """
```

**使用示例：**

```python
# 加载预训练模型
model = NexusForCausalLM.from_pretrained(
    "nexus-ai/nexus-7b",
    torch_dtype=torch.float16,
    device_map="auto",
)

# 前向传播（计算损失）
outputs = model(
    input_ids=tokenizer("Hello", return_tensors="pt").input_ids.cuda(),
    labels=tokenizer("Hello world", return_tensors="pt").input_ids.cuda(),
)
print(f"Loss: {outputs.loss.item():.4f}")

# 文本生成
output_ids = model.generate(
    input_ids=tokenizer("请解释什么是深度学习：", return_tensors="pt").input_ids.cuda(),
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

# 保存模型
model.save_pretrained("./my-model")
```

---

### NexusTokenizer

分词器，支持 BPE 分词和特殊 token 管理。

```python
from nexus_llm import NexusTokenizer
```

**类定义：**

```python
class NexusTokenizer:
    """Nexus-LLM 分词器。

    基于 BPE (Byte-Pair Encoding) 算法，支持中英文混合文本分词。
    """

    @classmethod
    def from_pretrained(cls, tokenizer_path: str) -> "NexusTokenizer":
        """从预训练分词器加载。"""

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], torch.Tensor]:
        """文本编码为 token ID。"""

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """token ID 解码为文本。"""

    def __call__(self, text, **kwargs):
        """支持 return_tensors="pt" 的便捷调用。"""
```

---

## 训练 API

### TrainingConfig

训练配置类。

```python
from nexus_llm.training import TrainingConfig
```

**类定义：**

```python
class TrainingConfig:
    """训练配置。

    参数:
        output_dir (str): 输出目录。默认: "./output"
        num_train_epochs (int): 训练轮数。默认: 3
        per_device_train_batch_size (int): 每设备训练批次大小。默认: 8
        per_device_eval_batch_size (int): 每设备评估批次大小。默认: 8
        gradient_accumulation_steps (int): 梯度累积步数。默认: 1
        learning_rate (float): 学习率。默认: 2e-5
        weight_decay (float): 权重衰减。默认: 0.01
        warmup_ratio (float): 学习率预热比例。默认: 0.03
        lr_scheduler_type (str): 学习率调度器类型。默认: "cosine"
        max_grad_norm (float): 梯度裁剪范数。默认: 1.0
        fp16 (bool): 启用 FP16。默认: False
        bf16 (bool): 启用 BF16。默认: False
        gradient_checkpointing (bool): 启用梯度检查点。默认: False
        logging_steps (int): 日志记录步数间隔。默认: 10
        save_steps (int): 模型保存步数间隔。默认: 500
        eval_steps (int): 评估步数间隔。默认: 500
        save_total_limit (int): 最多保存检查点数量。默认: 5
        dataloader_num_workers (int): 数据加载工作线程数。默认: 4
        seed (int): 随机种子。默认: 42
        dataloader_pin_memory (bool): 是否锁页内存。默认: True
    """
```

---

### Trainer

核心训练器，支持单机和分布式训练。

```python
from nexus_llm.training import Trainer
```

**类定义：**

```python
class Trainer:
    """Nexus-LLM 核心训练器。

    参数:
        model (NexusForCausalLM): 待训练模型。
        config (TrainingConfig): 训练配置。
        train_file (str): 训练数据文件路径（JSONL 格式）。
        eval_file (Optional[str]): 评估数据文件路径。
        tokenizer (Optional[NexusTokenizer]): 分词器。
        data_collator (Optional[Callable]): 数据整理函数。
    """

    def train(self, resume_from_checkpoint: Optional[str] = None) -> TrainOutput:
        """开始训练。

        参数:
            resume_from_checkpoint: 从指定检查点恢复训练。

        返回:
            TrainOutput: 训练结果，包含全局步数、训练损失等。
        """

    def evaluate(self, eval_dataset=None) -> Dict[str, float]:
        """运行评估。

        返回:
            Dict[str, float]: 评估指标字典。
        """

    def save_model(self, output_dir: Optional[str] = None):
        """保存模型和分词器。"""

    def save_checkpoint(self, output_dir: Optional[str] = None):
        """保存训练检查点（含优化器状态）。"""

    def push_to_hub(self, repo_id: str, private: bool = False):
        """上传模型到 HuggingFace Hub。"""

    def add_callback(self, callback: TrainerCallback):
        """添加训练回调。"""
```

**使用示例：**

```python
from nexus_llm import NexusForCausalLM, NexusTokenizer
from nexus_llm.training import Trainer, TrainingConfig

# 加载模型和分词器
model = NexusForCausalLM.from_pretrained("nexus-ai/nexus-7b")
tokenizer = NexusTokenizer.from_pretrained("nexus-ai/nexus-7b")

# 配置训练
config = TrainingConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=200,
    eval_steps=200,
)

# 创建训练器
trainer = Trainer(
    model=model,
    config=config,
    train_file="./data/train.jsonl",
    eval_file="./data/eval.jsonl",
    tokenizer=tokenizer,
)

# 自定义回调
from nexus_llm.training import TrainerCallback

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"Step {state.global_step}: loss={logs['loss']:.4f}")

trainer.add_callback(LoggingCallback())

# 开始训练
result = trainer.train()
print(f"训练完成！总步数: {result.global_step}")

# 保存并上传
trainer.save_model()
trainer.push_to_hub("my-org/nexus-7b-finetuned")
```

---

### DistributedTrainer

分布式训练器，支持数据并行、张量并行和流水线并行。

```python
from nexus_llm.training import DistributedTrainer, DistributedConfig
```

**类定义：**

```python
class DistributedConfig:
    """分布式训练配置。

    参数:
        strategy (str): 分布式策略。可选: "ddp", "deepspeed_stage2", "deepspeed_stage3"
        num_gpus (int): 使用的 GPU 数量。
        tensor_parallel_size (int): 张量并行大小。默认: 1
        pipeline_parallel_size (int): 流水线并行大小。默认: 1
        deepspeed_config (Optional[dict]): DeepSpeed 配置覆盖。
        fsdp_config (Optional[dict]): FSDP 配置覆盖。
    """


class DistributedTrainer(Trainer):
    """分布式训练器，继承自 Trainer。

    在 Trainer 基础上增加了多 GPU 分布式训练能力，
    支持 DDP、DeepSpeed ZeRO-2/3 和 FSDP。

    参数:
        model: 待训练模型。
        config: 训练配置。
        dist_config: 分布式配置。
        **kwargs: 其他 Trainer 参数。
    """

    def launch(self):
        """启动分布式训练。"""

    def get_world_size(self) -> int:
        """获取全局进程数。"""

    def get_rank(self) -> int:
        """获取当前进程排名。"""
```

**使用示例：**

```python
from nexus_llm.training import DistributedTrainer, DistributedConfig, TrainingConfig

dist_config = DistributedConfig(
    strategy="deepspeed_stage3",
    num_gpus=8,
)

training_config = TrainingConfig(
    output_dir="./output",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    fp16=True,
)

trainer = DistributedTrainer(
    model=model,
    config=training_config,
    dist_config=dist_config,
    train_file="./data/train.jsonl",
)

trainer.launch()
```

---

### FineTuningTrainer

微调训练器，内置 LoRA、QLoRA 和 P-Tuning 支持。

```python
from nexus_llm.training import FineTuningTrainer, LoRAConfig
```

**类定义：**

```python
class LoRAConfig:
    """LoRA 配置。

    参数:
        r (int): LoRA 秩。默认: 8
        alpha (int): LoRA 缩放因子。默认: 16
        dropout (float): LoRA Dropout。默认: 0.05
        target_modules (List[str]): 目标模块列表。默认: ["q_proj", "v_proj"]
        bias (str): 偏置处理方式。可选: "none", "all", "lora_only"。默认: "none"
        task_type (str): 任务类型。默认: "CAUSAL_LM"
    """


class FineTuningTrainer(Trainer):
    """微调训练器，支持 LoRA/QLoRA/P-Tuning。

    参数:
        model: 基础模型。
        config: 训练配置。
        lora_config: LoRA 配置（为 None 则全参数微调）。
        **kwargs: 其他 Trainer 参数。
    """

    def merge_and_save(self, output_dir: str):
        """合并 LoRA 权重并保存完整模型。"""

    def get_trainable_parameters(self) -> Dict[str, int]:
        """获取可训练参数统计。

        返回:
            {"trainable": int, "total": int, "percentage": float}
        """
```

**使用示例：**

```python
from nexus_llm.training import FineTuningTrainer, LoRAConfig, TrainingConfig

lora_config = LoRAConfig(
    r=16,
    alpha=32,
    dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

trainer = FineTuningTrainer(
    model=model,
    config=TrainingConfig(output_dir="./output", num_train_epochs=3, fp16=True),
    lora_config=lora_config,
    train_file="./data/train.jsonl",
)

# 查看可训练参数
params = trainer.get_trainable_parameters()
print(f"可训练参数: {params['trainable']:,} / {params['total']:,} ({params['percentage']:.2f}%)")

trainer.train()

# 合并 LoRA 权重并保存
trainer.merge_and_save("./output/merged-model")
```

---

## 推理服务 API

### InferenceEngine

高性能推理引擎，支持连续批处理和 KV Cache 优化。

```python
from nexus_llm.serving import InferenceEngine
```

**类定义：**

```python
class InferenceEngine:
    """推理引擎。

    参数:
        model (NexusForCausalLM): 模型实例。
        tokenizer (NexusTokenizer): 分词器。
        max_new_tokens (int): 最大生成 token 数。默认: 512
        temperature (float): 采样温度。默认: 1.0
        top_p (float): nucleus 采样阈值。默认: 1.0
        top_k (int): top-k 采样参数。默认: 50
        repetition_penalty (float): 重复惩罚。默认: 1.0
        max_batch_size (int): 最大批处理大小。默认: 32
        gpu_memory_utilization (float): GPU 显存利用率。默认: 0.9
    """

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本。

        参数:
            prompt: 输入提示文本。
            **kwargs: 覆盖默认生成参数。

        返回:
            str: 生成的文本。
        """

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """对话生成。

        参数:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]。
            **kwargs: 覆盖默认生成参数。

        返回:
            str: 助手回复文本。
        """

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成文本。

        参数:
            prompt: 输入提示文本。
            **kwargs: 覆盖默认生成参数。

        返回:
            Iterator[str]: 逐 token 生成文本的迭代器。
        """

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """流式对话生成。

        返回:
            Iterator[str]: 逐 token 生成回复的迭代器。
        """

    def embed(self, text: str) -> List[float]:
        """获取文本嵌入向量。

        返回:
            List[float]: 嵌入向量。
        """

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量。"""

    def warmup(self):
        """预热模型（首次推理前调用以优化性能）。"""
```

---

### ModelServer

模型推理服务器，提供 HTTP API。

```python
from nexus_llm.serving import ModelServer
```

**类定义：**

```python
class ModelServer:
    """模型推理服务器。

    参数:
        model_path (str): 模型路径。
        host (str): 监听地址。默认: "0.0.0.0"
        port (int): 监听端口。默认: 8000
        gpu_memory_utilization (float): GPU 显存利用率。默认: 0.9
        max_model_len (int): 模型最大上下文长度。默认: 4096
        max_num_seqs (int): 最大并发序列数。默认: 256
        api_key (Optional[str]): API 密钥。
        ssl_certfile (Optional[str]): SSL 证书路径。
        ssl_keyfile (Optional[str]): SSL 密钥路径。
    """

    def start(self, blocking: bool = True):
        """启动服务器。

        参数:
            blocking: 是否阻塞主线程。
        """

    def stop(self):
        """停止服务器。"""

    def get_health(self) -> Dict:
        """获取服务健康状态。"""

    def get_stats(self) -> Dict:
        """获取服务统计信息（请求数、延迟等）。"""
```

**使用示例：**

```python
from nexus_llm.serving import ModelServer

server = ModelServer(
    model_path="nexus-ai/nexus-7b",
    host="0.0.0.0",
    port=8000,
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    api_key="sk-nexus-12345",
)

# 非阻塞启动
server.start(blocking=False)

# 检查状态
print(server.get_health())
print(server.get_stats())

# 停止服务
server.stop()
```

---

### OpenAI 兼容 API

Nexus-LLM 推理服务提供与 OpenAI API 完全兼容的 HTTP 接口。

#### POST /v1/chat/completions

创建聊天补全。

**请求：**

```json
{
  "model": "nexus-7b",
  "messages": [
    {"role": "system", "content": "你是一个有帮助的AI助手。"},
    {"role": "user", "content": "你好！"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 2048,
  "stream": false,
  "stop": ["\n"],
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "n": 1
}
```

**响应：**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "nexus-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！很高兴见到你。有什么我可以帮助你的吗？"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 15,
    "total_tokens": 35
  }
}
```

**流式响应（stream=true）：**

```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"role":"assistant","content":"你"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"好"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"！"},"index":0}]}

data: [DONE]
```

#### GET /v1/models

列出可用模型。

**响应：**

```json
{
  "object": "list",
  "data": [
    {
      "id": "nexus-7b",
      "object": "model",
      "created": 1700000000,
      "owned_by": "nexus-ai",
      "permission": []
    }
  ]
}
```

#### POST /v1/completions

创建文本补全（兼容 OpenAI Completions API）。

**请求：**

```json
{
  "model": "nexus-7b",
  "prompt": "从前有座山，",
  "max_tokens": 256,
  "temperature": 0.8,
  "stop": ["\n\n"]
}
```

#### POST /v1/embeddings

获取文本嵌入向量。

**请求：**

```json
{
  "model": "nexus-7b",
  "input": "这是一段需要嵌入的文本"
}
```

**响应：**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0023, -0.0094, 0.0156, "..."],
      "index": 0
    }
  ],
  "model": "nexus-7b",
  "usage": {"prompt_tokens": 12, "total_tokens": 12}
}
```

#### GET /health

健康检查端点。

**响应：**

```json
{
  "status": "healthy",
  "model": "nexus-7b",
  "gpu_memory_used": 14.2,
  "gpu_memory_total": 24.0,
  "uptime_seconds": 3600
}
```

#### GET /v1/stats

服务统计信息。

**响应：**

```json
{
  "total_requests": 1024,
  "active_requests": 3,
  "avg_latency_ms": 45.2,
  "p99_latency_ms": 120.5,
  "tokens_generated": 512000,
  "gpu_utilization": 0.87
}
```

---

## RLHF API

### RewardModel

奖励模型，用于评估生成文本的质量。

```python
from nexus_llm.rlhf import RewardModel, RewardConfig
```

**类定义：**

```python
class RewardConfig(NexusConfig):
    """奖励模型配置。

    继承自 NexusConfig，增加以下参数:

    参数:
        num_pooled_outputs (int): 池化输出数量。默认: 1
        reward_head_hidden_size (int): 奖励头隐藏层维度。默认: None（使用 hidden_size）
        dropout (float): Dropout 概率。默认: 0.1
    """


class RewardModel(nn.Module):
    """奖励模型。

    基于 NexusModel 架构，在最后一层隐藏状态上添加线性奖励头，
    输出一个标量奖励分数。

    参数:
        config (RewardConfig): 奖励模型配置。
        base_model (Optional[NexusModel]): 基础模型（可选，用于初始化）。
    """

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """计算奖励分数。

        参数:
            input_ids: 输入 token ID。
            attention_mask: 注意力掩码。

        返回:
            torch.FloatTensor: 奖励分数，形状为 (batch_size,)
        """

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "RewardModel":
        """从预训练权重加载奖励模型。"""

    def save_pretrained(self, output_dir: str):
        """保存奖励模型。"""
```

**使用示例：**

```python
from nexus_llm.rlhf import RewardModel

# 加载预训练的奖励模型
reward_model = RewardModel.from_pretrained("nexus-ai/nexus-7b-reward")

# 评估两个回答的质量
good_response = "机器学习是人工智能的一个分支，通过数据驱动的方式..."
bad_response = "我不知道什么是机器学习。"

score_good = reward_model(good_response)
score_bad = reward_model(bad_response)

print(f"优质回答分数: {score_good.item():.4f}")
print(f"低质回答分数: {score_bad.item():.4f}")
```

---

### PPOTrainer

PPO 训练器，用于基于人类反馈的强化学习。

```python
from nexus_llm.rlhf import PPOTrainer, PPOConfig
```

**类定义：**

```python
class PPOConfig:
    """PPO 训练配置。

    参数:
        learning_rate (float): 策略模型学习率。默认: 1e-6
        kl_coef (float): KL 散度惩罚系数。默认: 0.1
        gamma (float): 折扣因子。默认: 1.0
        lam (float): GAE lambda。默认: 0.95
        cliprange (float): PPO 裁剪范围。默认: 0.2
        cliprange_value (float): 价值函数裁剪范围。默认: 0.2
        vf_coef (float): 价值函数损失系数。默认: 0.1
        entropy_coef (float): 熵正则化系数。默认: 0.01
        ppo_epochs (int): 每次 PPO 更新的 epoch 数。默认: 4
        mini_batch_size (int): PPO mini-batch 大小。默认: 256
        gradient_accumulation_steps (int): 梯度累积步数。默认: 1
        max_new_tokens (int): 生成最大 token 数。默认: 256
        temperature (float): 生成温度。默认: 0.7
        early_stopping (bool): 是否启用早停。默认: True
        target_kl (float): 目标 KL 散度（早停阈值）。默认: 0.1
    """


class PPOTrainer:
    """PPO 训练器。

    参数:
        config (PPOConfig): PPO 配置。
        policy_model (NexusForCausalLM): 策略模型（待优化）。
        reference_model (NexusForCausalLM): 参考模型（固定不更新）。
        reward_model (RewardModel): 奖励模型。
        value_model (Optional[NexusForCausalLM]): 价值模型（可选，默认共享策略模型）。
        tokenizer (NexusTokenizer): 分词器。
    """

    def train(self, dataloader) -> Dict[str, float]:
        """执行一轮 PPO 训练。

        参数:
            dataloader: 提示数据加载器。

        返回:
            Dict[str, float]: 训练指标，包含:
                - policy_loss: 策略损失
                - value_loss: 价值函数损失
                - reward_mean: 平均奖励
                - kl_divergence: KL 散度
                - entropy: 策略熵
        """

    def generate(self, prompts: List[str]) -> List[str]:
        """使用当前策略生成回答。"""

    def step(self, prompts: List[str], responses: List[str]) -> Dict[str, float]:
        """执行单步 PPO 更新。

        参数:
            prompts: 提示列表。
            responses: 对应的回答列表。

        返回:
            Dict[str, float]: 训练指标。
        """

    def log_stats(self):
        """记录训练统计信息。"""
```

**使用示例：**

```python
from nexus_llm import NexusForCausalLM, NexusTokenizer
from nexus_llm.rlhf import PPOTrainer, PPOConfig, RewardModel

# 加载模型
policy_model = NexusForCausalLM.from_pretrained("nexus-ai/nexus-7b-sft")
reference_model = NexusForCausalLM.from_pretrained("nexus-ai/nexus-7b-sft")
reward_model = RewardModel.from_pretrained("nexus-ai/nexus-7b-reward")
tokenizer = NexusTokenizer.from_pretrained("nexus-ai/nexus-7b")

# 配置 PPO
ppo_config = PPOConfig(
    learning_rate=1e-6,
    kl_coef=0.1,
    ppo_epochs=4,
    mini_batch_size=128,
    max_new_tokens=256,
    temperature=0.7,
)

# 初始化训练器
trainer = PPOTrainer(
    config=ppo_config,
    policy_model=policy_model,
    reference_model=reference_model,
    reward_model=reward_model,
    tokenizer=tokenizer,
)

# 训练
for epoch in range(num_epochs):
    stats = trainer.step(prompts, responses)
    print(f"Epoch {epoch}: reward={stats['reward_mean']:.4f}, kl={stats['kl_divergence']:.4f}")
```

---

### RLHFConfig

RLHF 流水线配置。

```python
from nexus_llm.rlhf import RLHFConfig

config = RLHFConfig(
    sft_model_path="nexus-ai/nexus-7b-sft",
    reward_model_path="nexus-ai/nexus-7b-reward",
    output_dir="./output/rlhf",
    ppo_config=PPOConfig(
        learning_rate=1e-6,
        kl_coef=0.1,
    ),
)
```

---

## Agent API

### ReActAgent

基于 ReAct（Reasoning + Acting）范式的智能体。

```python
from nexus_llm.agent import ReActAgent, AgentConfig
```

**类定义：**

```python
class AgentConfig:
    """Agent 配置。

    参数:
        model_path (str): 模型路径。
        max_iterations (int): 最大推理迭代次数。默认: 10
        max_tokens_per_step (int): 每步最大 token 数。默认: 512
        temperature (float): 生成温度。默认: 0.7
        verbose (bool): 是否打印推理过程。默认: False
        return_intermediate_steps (bool): 是否返回中间步骤。默认: False
    """


class ReActAgent:
    """ReAct 智能体。

    通过交替执行"思考-行动-观察"循环来解决复杂任务。

    参数:
        config (AgentConfig): Agent 配置。
        tools (List[BaseTool]): 可用工具列表。
        system_prompt (Optional[str]): 系统提示词。
    """

    def run(self, task: str) -> AgentOutput:
        """执行任务。

        参数:
            task: 任务描述。

        返回:
            AgentOutput: 包含最终答案和中间步骤。
        """

    def chat(self, message: str, history: List[Dict] = None) -> str:
        """对话模式执行。

        参数:
            message: 用户消息。
            history: 对话历史。

        返回:
            str: Agent 回复。
        """

    def add_tool(self, tool: BaseTool):
        """添加工具。"""

    def remove_tool(self, tool_name: str):
        """移除工具。"""

    def list_tools(self) -> List[str]:
        """列出可用工具名称。"""
```

**使用示例：**

```python
from nexus_llm.agent import ReActAgent, AgentConfig
from nexus_llm.agent.tools import SearchTool, CalculatorTool

# 创建 Agent
agent = ReActAgent(
    config=AgentConfig(
        model_path="nexus-ai/nexus-7b",
        max_iterations=5,
        verbose=True,
    ),
    tools=[SearchTool(), CalculatorTool()],
    system_prompt="你是一个有帮助的AI助手，可以使用工具来回答问题。",
)

# 执行任务
result = agent.run("2024年中国的GDP是多少？换算成美元是多少？")
print(f"最终答案: {result.answer}")
print(f"中间步骤: {result.intermediate_steps}")
```

---

### FunctionCallingAgent

基于函数调用的智能体。

```python
from nexus_llm.agent import FunctionCallingAgent
```

**类定义：**

```python
class FunctionCallingAgent:
    """函数调用 Agent。

    通过模型原生函数调用能力来使用工具，
    相比 ReAct 更加高效和准确。

    参数:
        config (AgentConfig): Agent 配置。
        tools (List[BaseTool]): 可用工具列表。
        system_prompt (Optional[str]): 系统提示词。
    """

    def run(self, task: str) -> AgentOutput:
        """执行任务。"""

    def chat(self, message: str, history: List[Dict] = None) -> str:
        """对话模式执行。"""

    def register_function(self, name: str, description: str, parameters: dict, handler: Callable):
        """注册自定义函数。

        参数:
            name: 函数名称。
            description: 函数描述（供模型理解）。
            parameters: JSON Schema 格式的参数定义。
            handler: 函数处理逻辑。
        """
```

**使用示例：**

```python
from nexus_llm.agent import FunctionCallingAgent, AgentConfig

agent = FunctionCallingAgent(
    config=AgentConfig(model_path="nexus-ai/nexus-7b"),
    system_prompt="你是一个天气助手。",
)

# 注册自定义函数
agent.register_function(
    name="get_weather",
    description="获取指定城市的当前天气信息",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
        },
        "required": ["city"],
    },
    handler=lambda city, unit="celsius": f"{city}: 25{unit[0]}, 晴天",
)

result = agent.run("北京今天天气怎么样？")
print(result.answer)
```

---

### ToolRegistry

工具注册中心，管理所有可用工具。

```python
from nexus_llm.agent import ToolRegistry
```

**类定义：**

```python
class ToolRegistry:
    """工具注册中心。

    管理工具的注册、发现和调用。
    """

    def register(self, tool: BaseTool):
        """注册工具。"""

    def unregister(self, tool_name: str):
        """注销工具。"""

    def get(self, tool_name: str) -> BaseTool:
        """获取工具实例。"""

    def list_tools(self) -> List[Dict]:
        """列出所有已注册工具的信息。"""

    def search(self, query: str) -> List[BaseTool]:
        """根据描述搜索工具。"""

    @classmethod
    def from_default(cls) -> "ToolRegistry":
        """创建包含所有内置工具的注册中心。"""
```

---

### BaseTool

工具基类，用于创建自定义工具。

```python
from nexus_llm.agent import BaseTool
```

**类定义：**

```python
class BaseTool(ABC):
    """工具基类。

    所有自定义工具必须继承此类并实现 execute 方法。
    """

    name: str = ""
    description: str = ""
    parameters_schema: dict = {}

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """执行工具逻辑。

        参数:
            **kwargs: 工具参数。

        返回:
            Any: 工具执行结果。
        """

    def to_openai_function(self) -> dict:
        """转换为 OpenAI 函数调用格式。"""

    def validate_args(self, **kwargs) -> bool:
        """验证参数是否合法。"""
```

**自定义工具示例：**

```python
from nexus_llm.agent import BaseTool

class DatabaseQueryTool(BaseTool):
    name = "database_query"
    description = "查询数据库并返回结果"
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "SQL 查询语句"},
            "database": {"type": "string", "description": "数据库名称"},
        },
        "required": ["query", "database"],
    }

    def __init__(self, db_connection):
        self.db = db_connection

    def execute(self, query: str, database: str) -> str:
        """执行 SQL 查询并返回结果。"""
        cursor = self.db.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        return str(results)
```

---

## 压缩 API

### KnowledgeDistiller

知识蒸馏工具，将大模型知识迁移到小模型。

```python
from nexus_llm.compression import KnowledgeDistiller, DistillConfig
```

**类定义：**

```python
class DistillConfig:
    """知识蒸馏配置。

    参数:
        teacher_model_path (str): 教师模型路径。
        student_config (NexusConfig): 学生模型配置。
        temperature (float): 蒸馏温度。默认: 4.0
        alpha (float): 蒸馏损失权重（0~1）。默认: 0.5
        intermediate_loss_weight (float): 中间层损失权重。默认: 0.25
        num_layers_to_match (int): 匹配的中间层数。默认: None（全部匹配）
    """


class KnowledgeDistiller:
    """知识蒸馏器。

    参数:
        config (DistillConfig): 蒸馏配置。
    """

    def distill(
        self,
        train_dataloader,
        eval_dataloader=None,
        num_epochs: int = 3,
    ) -> NexusForCausalLM:
        """执行知识蒸馏。

        返回:
            NexusForCausalLM: 蒸馏后的学生模型。
        """

    def evaluate(self, dataloader) -> Dict[str, float]:
        """评估学生模型性能。"""
```

**使用示例：**

```python
from nexus_llm.compression import KnowledgeDistiller, DistillConfig
from nexus_llm import NexusConfig

student_config = NexusConfig(
    hidden_size=2048,
    num_hidden_layers=16,
    num_attention_heads=16,
    intermediate_size=5504,
)

distill_config = DistillConfig(
    teacher_model_path="nexus-ai/nexus-7b",
    student_config=student_config,
    temperature=4.0,
    alpha=0.5,
)

distiller = KnowledgeDistiller(distill_config)
student_model = distiller.distill(train_dataloader, num_epochs=5)
student_model.save_pretrained("./output/nexus-3b-distilled")
```

---

### StructuredPruner

结构化剪枝工具。

```python
from nexus_llm.compression import StructuredPruner, PruneConfig
```

**类定义：**

```python
class PruneConfig:
    """剪枝配置。

    参数:
        pruning_method (str): 剪枝方法。可选: "magnitude", "taylor", "l0"
        pruning_ratio (float): 剪枝比例。默认: 0.3
        target_modules (List[str]): 目标模块。默认: ["q_proj", "v_proj"]
        pruning_schedule (str): 剪枝调度。可选: "one_shot", "iterative", "gradual"
        num_pruning_steps (int): 渐进剪枝步数。默认: 5
        importance_criteria (str): 重要性标准。默认: "magnitude"
    """


class StructuredPruner:
    """结构化剪枝器。

    参数:
        model (NexusForCausalLM): 待剪枝模型。
        config (PruneConfig): 剪枝配置。
    """

    def prune(self) -> NexusForCausalLM:
        """执行剪枝。

        返回:
            NexusForCausalLM: 剪枝后的模型。
        """

    def analyze(self) -> Dict[str, Dict]:
        """分析各层的剪枝敏感度。

        返回:
            Dict: 各模块的重要性分数和推荐剪枝比例。
        """

    def get_sparsity(self) -> Dict[str, float]:
        """获取当前模型的稀疏度统计。"""
```

---

### ModelCompressor

模型压缩流水线，集成蒸馏、剪枝和量化。

```python
from nexus_llm.compression import ModelCompressor, CompressionPipeline
```

**类定义：**

```python
class CompressionPipeline:
    """压缩流水线配置。

    参数:
        steps (List[str]): 压缩步骤列表。可选: "distill", "prune", "quantize"
        distill_config (Optional[DistillConfig]): 蒸馏配置。
        prune_config (Optional[PruneConfig]): 剪枝配置。
        quantize_config (Optional[QuantizeConfig]): 量化配置。
    """


class ModelCompressor:
    """模型压缩器。

    参数:
        pipeline (CompressionPipeline): 压缩流水线配置。
    """

    def compress(self, model: NexusForCausalLM, dataloader) -> NexusForCausalLM:
        """执行完整的压缩流水线。

        返回:
            NexusForCausalLM: 压缩后的模型。
        """

    def benchmark(
        self,
        original_model: NexusForCausalLM,
        compressed_model: NexusForCausalLM,
        eval_dataloader,
    ) -> Dict[str, Any]:
        """对比原始模型和压缩模型的性能。

        返回:
            Dict: 包含大小、速度、精度等对比指标。
        """
```

---

## 增量学习 API

### IncrementalTrainer

增量学习训练器，支持持续学习而不遗忘旧知识。

```python
from nexus_llm.incremental import IncrementalTrainer, IncrementalConfig
```

**类定义：**

```python
class IncrementalConfig(TrainingConfig):
    """增量学习配置。

    继承 TrainingConfig，增加以下参数:

    参数:
        regularizer_type (str): 正则化方法。可选: "ewc", "l2", "si", "mas"。默认: "ewc"
        ewc_lambda (float): EWC 正则化强度。默认: 5000.0
        replay_buffer_size (int): 回放缓冲区大小。默认: 0（不使用回放）
        replay_ratio (float): 回放样本比例。默认: 0.1
        adapter_type (str): 适配器类型。可选: "lora", "adapter", "prompt"。默认: "lora"
        eval_old_tasks (bool): 是否在旧任务上评估。默认: True
    """


class IncrementalTrainer(FineTuningTrainer):
    """增量学习训练器。

    在 FineTuningTrainer 基础上增加了灾难性遗忘防护能力，
    支持 EWC、SI 等正则化方法和经验回放。

    参数:
        model: 基础模型。
        config: 增量学习配置。
        **kwargs: 其他参数。
    """

    def train_on_task(self, task_name: str, train_dataloader) -> Dict[str, float]:
        """在新任务上训练。

        参数:
            task_name: 任务名称。
            train_dataloader: 任务数据加载器。

        返回:
            Dict[str, float]: 训练指标。
        """

    def evaluate_all_tasks(self) -> Dict[str, float]:
        """在所有已学任务上评估。

        返回:
            Dict[str, float]: 各任务的评估指标。
        """

    def compute_forgetting(self) -> Dict[str, float]:
        """计算各任务的遗忘程度。

        返回:
            Dict[str, float]: 各任务性能下降值。
        """

    def save_task_checkpoint(self, task_name: str, output_dir: str):
        """保存任务检查点。"""

    def load_task_checkpoint(self, checkpoint_dir: str):
        """加载任务检查点。"""
```

**使用示例：**

```python
from nexus_llm.incremental import IncrementalTrainer, IncrementalConfig

config = IncrementalConfig(
    output_dir="./output/incremental",
    regularizer_type="ewc",
    ewc_lambda=5000.0,
    replay_buffer_size=1000,
    adapter_type="lora",
    num_train_epochs=3,
    fp16=True,
)

trainer = IncrementalTrainer(
    model=model,
    config=config,
)

# 依次在不同任务上训练
for task_name, task_data in tasks:
    stats = trainer.train_on_task(task_name, task_data)
    print(f"Task {task_name}: loss={stats['loss']:.4f}")

    # 检查遗忘情况
    forgetting = trainer.compute_forgetting()
    print(f"Forgetting: {forgetting}")

    # 保存检查点
    trainer.save_task_checkpoint(task_name, f"./checkpoints/{task_name}")
```

---

### EWCRegularizer

弹性权重巩固（Elastic Weight Consolidation）正则化器。

```python
from nexus_llm.incremental import EWCRegularizer
```

**类定义：**

```python
class EWCRegularizer:
    """EWC 正则化器。

    通过计算 Fisher 信息矩阵来估计参数重要性，
    在新任务训练时对重要参数施加更大的正则化。

    参数:
        model (nn.Module): 模型。
        lambda_ (float): 正则化强度。默认: 5000.0
        mode (str): Fisher 信息计算模式。可选: "online", "offline"。默认: "online"
    """

    def compute_fisher(self, dataloader):
        """计算 Fisher 信息矩阵。

        参数:
            dataloader: 数据加载器。
        """

    def penalty(self) -> torch.Tensor:
        """计算 EWC 正则化损失。

        返回:
            torch.Tensor: EWC 损失标量。
        """

    def update(self, model: nn.Module):
        """在任务切换后更新正则化器状态。"""

    def save(self, path: str):
        """保存 Fisher 信息矩阵。"""

    def load(self, path: str):
        """加载 Fisher 信息矩阵。"""
```

---

### AdapterModule

适配器模块，支持 LoRA、Adapter 和 Prefix Tuning。

```python
from nexus_llm.incremental import AdapterModule, AdapterConfig
```

**类定义：**

```python
class AdapterConfig:
    """适配器配置。

    参数:
        adapter_type (str): 适配器类型。可选: "lora", "houlsby", "prefix", "p_tuning"
        hidden_dim (int): 适配器隐藏层维度。默认: 64
        rank (int): LoRA 秩。默认: 8
        num_layers (int): Prefix Tuning 虚拟 token 数。默认: 10
        activation (str): 激活函数。默认: "gelu"
        dropout (float): Dropout。默认: 0.1
    """


class AdapterModule(nn.Module):
    """适配器模块。

    可以插入到 Transformer 层中，实现参数高效的增量学习。

    参数:
        config (AdapterConfig): 适配器配置。
        input_dim (int): 输入维度。
        output_dim (int): 输出维度。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""

    def merge(self) -> None:
        """将适配器权重合并到基础权重中（仅 LoRA 支持）。"""

    def unmerge(self) -> None:
        """取消合并。"""

    def get_num_parameters(self) -> int:
        """获取适配器参数数量。"""
```

**使用示例：**

```python
from nexus_llm.incremental import AdapterModule, AdapterConfig

# 创建适配器
adapter_config = AdapterConfig(adapter_type="lora", rank=16)
adapter = AdapterModule(adapter_config, input_dim=4096, output_dim=4096)

# 查看参数量
print(f"适配器参数量: {adapter.get_num_parameters():,}")

# 合并到基础模型
adapter.merge()
```

---

> 更多使用示例请参阅 [快速入门](./getting_started.md) 和 [架构设计](./architecture.md)。
