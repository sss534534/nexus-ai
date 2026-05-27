# Nexus-LLM RLHF 训练指南

> 本文档详细介绍如何使用 Nexus-LLM 框架完成 RLHF（基于人类反馈的强化学习）全流程训练，包括监督微调（SFT）、奖励模型训练和 PPO 对齐训练。

---

## 目录

- [RLHF 流水线概览](#rlhf-流水线概览)
- [第一步：监督微调 (SFT)](#第一步监督微调-sft)
  - [数据准备](#数据准备)
  - [SFT 训练配置](#sft-训练配置)
  - [运行 SFT 训练](#运行-sft-训练)
  - [评估 SFT 模型](#评估-sft-模型)
- [第二步：奖励模型训练](#第二步奖励模型训练)
  - [偏好数据准备](#偏好数据准备)
  - [奖励模型配置](#奖励模型配置)
  - [训练奖励模型](#训练奖励模型)
  - [评估奖励模型](#评估奖励模型)
- [第三步：PPO 训练](#第三步ppo-训练)
  - [PPO 原理简介](#ppo-原理简介)
  - [PPO 配置](#ppo-配置)
  - [运行 PPO 训练](#运行-ppo-训练)
  - [监控训练过程](#监控训练过程)
- [配置参考](#配置参考)
  - [SFTConfig](#sftconfig)
  - [RewardModelConfig](#rewardmodelconfig)
  - [PPOConfig](#ppoconfig)
- [最佳实践](#最佳实践)
- [故障排除](#故障排除)

---

## RLHF 流水线概览

RLHF（Reinforcement Learning from Human Feedback）通过人类偏好反馈来对齐模型输出与人类期望，是训练高质量对话模型的核心技术。

```
+=====================================================================+
|                      RLHF 训练流水线                                 |
+=====================================================================+
                                                                     |
  阶段 1: 监督微调 (SFT)                                              |
  +------------------+     +------------------+     +--------------+ |
  | 指令数据集        |---->| 基础模型          |---->| SFT 模型      | |
  | (instruction,     |     | (Nexus-7B-Base)  |     | (对齐指令格式)| |
  |  output)          |     |                  |     |              | |
  +------------------+     +------------------+     +--------------+ |
                                                                     |
  阶段 2: 奖励模型训练 (RM)                                           |
  +------------------+     +------------------+     +--------------+ |
  | 偏好数据集        |---->| SFT 模型          |---->| 奖励模型      | |
  | (chosen,          |     | (冻结)           |     | (评分函数)    | |
  |  rejected)        |     |                  |     |              | |
  +------------------+     +------------------+     +--------------+ |
                                                                     |
  阶段 3: PPO 对齐训练                                                |
  +------------------+     +------------------+     +--------------+ |
  | 提示数据集        |---->| SFT 模型          |---->| 对齐模型      | |
  | (prompt)          |     | (策略模型,可更新)  |     | (PPO 优化后)  | |
  +------------------+     +------------------+     +--------------+ |
                              |               |                      |
                              v               v                      |
                         +----------+   +----------+                 |
                         | 参考模型  |   | 奖励模型  |                 |
                         | (冻结)   |   | (冻结)   |                 |
                         +----------+   +----------+                 |
                                                                     |
+=====================================================================+
```

### 各阶段目标

| 阶段 | 输入 | 输出 | 目标 |
|------|------|------|------|
| SFT | 指令-回答对 | SFT 模型 | 学会遵循指令格式 |
| RM | 偏好对比对 | 奖励模型 | 学会评判回答质量 |
| PPO | 提示 + 奖励信号 | 对齐模型 | 生成人类偏好的回答 |

---

## 第一步：监督微调 (SFT)

SFT 阶段使用高质量的指令-回答对数据，让基础模型学会遵循指令并生成结构化的回答。

### 数据准备

SFT 数据为 JSONL 格式，每行包含一条指令-回答对：

```jsonl
{"instruction": "请解释什么是机器学习", "output": "机器学习是人工智能的一个重要分支，它使计算机系统能够从数据中学习并改进，而无需显式编程。机器学习的核心思想是通过算法分析数据，识别模式，并做出预测或决策。\n\n主要类型包括：\n1. 监督学习：使用标注数据训练模型\n2. 无监督学习：从未标注数据中发现模式\n3. 强化学习：通过试错和奖励信号学习"}
{"instruction": "用Python实现快速排序", "output": "```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n\n# 使用示例\narr = [3, 6, 8, 10, 1, 2, 1]\nprint(quicksort(arr))  # [1, 1, 2, 3, 6, 8, 10]\n```"}
{"instruction": "写一首关于秋天的诗", "output": "秋风起处叶纷飞，\n金桂飘香满院扉。\n远山层林染红透，\n一江秋水映余晖。"}
```

**数据质量要求：**

| 要求 | 说明 |
|------|------|
| 多样性 | 覆盖多种任务类型（问答、写作、编程、推理等） |
| 质量 | 回答准确、完整、格式规范 |
| 长度 | 回答长度适中，避免过短或过长 |
| 安全性 | 不包含有害、偏见或错误信息 |
| 数量 | 建议至少 10K 条高质量数据 |

**数据预处理脚本：**

```python
"""prepare_sft_data.py - SFT 数据预处理"""
import json
from nexus_llm import NexusTokenizer

tokenizer = NexusTokenizer.from_pretrained("nexus-ai/nexus-7b")

def process_sft_data(input_file: str, output_file: str, max_length: int = 2048):
    """处理 SFT 数据，添加特殊 token 并截断。"""
    processed = []
    skipped = 0

    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            instruction = data["instruction"]
            output = data["output"]

            # 构建对话格式
            prompt = f"### 指令:\n{instruction}\n\n### 回答:\n{output}"
            tokens = tokenizer.encode(prompt)

            if len(tokens) > max_length:
                skipped += 1
                continue

            # 构建标签（仅对 output 部分计算损失）
            prompt_tokens = tokenizer.encode(f"### 指令:\n{instruction}\n\n### 回答:\n")
            labels = [-100] * len(prompt_tokens) + tokens[len(prompt_tokens):]

            processed.append({
                "input_ids": tokens,
                "labels": labels,
                "attention_mask": [1] * len(tokens),
            })

    # 保存处理后的数据
    with open(output_file, 'w') as f:
        for item in processed:
            f.write(json.dumps(item) + '\n')

    print(f"处理完成: {len(processed)} 条有效, {skipped} 条跳过")

if __name__ == "__main__":
    process_sft_data("./data/raw_sft.jsonl", "./data/sft_processed.jsonl")
```

### SFT 训练配置

```python
"""sft_train.py - SFT 训练脚本"""
from nexus_llm import NexusForCausalLM, NexusTokenizer
from nexus_llm.training import FineTuningTrainer, LoRAConfig, TrainingConfig

# 模型和分词器
model = NexusForCausalLM.from_pretrained("nexus-ai/nexus-7b-base")
tokenizer = NexusTokenizer.from_pretrained("nexus-ai/nexus-7b-base")

# LoRA 配置（推荐，节省显存）
lora_config = LoRAConfig(
    r=16,
    alpha=32,
    dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# 训练配置
training_config = TrainingConfig(
    output_dir="./output/sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,     # 有效 batch size = 4 * 8 = 32
    learning_rate=2e-5,
    warmup_ratio=0.03,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=200,
    eval_steps=200,
    save_total_limit=3,
    dataloader_num_workers=4,
    seed=42,
)

# 创建训练器
trainer = FineTuningTrainer(
    model=model,
    config=training_config,
    lora_config=lora_config,
    train_file="./data/sft_processed.jsonl",
    eval_file="./data/sft_eval.jsonl",
    tokenizer=tokenizer,
)

# 查看可训练参数
params = trainer.get_trainable_parameters()
print(f"可训练参数: {params['trainable']:,} / {params['total']:,} ({params['percentage']:.2f}%)")

# 开始训练
trainer.train()

# 合并 LoRA 权重并保存
trainer.merge_and_save("./output/sft/merged-model")
```

### 运行 SFT 训练

```bash
# 使用 Python 脚本
python sft_train.py

# 使用命令行工具
nexus-train \
  --model-path nexus-ai/nexus-7b-base \
  --train-file ./data/sft_processed.jsonl \
  --eval-file ./data/sft_eval.jsonl \
  --output-dir ./output/sft \
  --num-epochs 3 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 \
  --fp16 \
  --gradient-checkpointing \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-target-modules q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj

# 多 GPU 训练
nexus-train \
  --model-path nexus-ai/nexus-7b-base \
  --train-file ./data/sft_processed.jsonl \
  --output-dir ./output/sft \
  --num-gpus 4 \
  --strategy deepspeed_stage3 \
  --num-epochs 3 \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --fp16
```

### 评估 SFT 模型

```bash
# 运行基准评估
nexus-eval \
  --model-path ./output/sft/merged-model \
  --tasks mmlu,ceval,humaneval,gsm8k \
  --output-dir ./eval_results/sft

# 交互式测试
python -c "
from nexus_llm import NexusForCausalLM, NexusTokenizer
from nexus_llm.serving import InferenceEngine

model = NexusForCausalLM.from_pretrained('./output/sft/merged-model')
tokenizer = NexusTokenizer.from_pretrained('./output/sft/merged-model')
engine = InferenceEngine(model=model, tokenizer=tokenizer, max_new_tokens=256)

while True:
    prompt = input('用户: ')
    if prompt == 'quit':
        break
    response = engine.generate(prompt)
    print(f'助手: {response}')
    print()
"
```

---

## 第二步：奖励模型训练

奖励模型学习人类对回答质量的偏好，为 PPO 训练提供奖励信号。

### 偏好数据准备

偏好数据为 JSONL 格式，每行包含一个提示和两个回答（偏好/不偏好）：

```jsonl
{"prompt": "请解释什么是量子纠缠", "chosen": "量子纠缠是量子力学中的一种现象，当两个或多个粒子发生纠缠后，无论它们相距多远，对其中一个粒子的测量会瞬间影响另一个粒子的状态。这种关联性超出了经典物理学的解释范围。\n\n爱因斯坦曾将这种现象称为'鬼魅般的超距作用'。量子纠缠是量子计算和量子通信的基础技术之一。", "rejected": "量子纠缠就是两个粒子有关系。"}
{"prompt": "用Python写一个二分查找", "chosen": "```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```\n\n时间复杂度 O(log n)，空间复杂度 O(1)。", "rejected": "二分查找就是找中间的。"}
```

**数据质量要求：**

| 要求 | 说明 |
|------|------|
| 对比性 | chosen 和 rejected 之间有明确的质量差异 |
| 一致性 | 相同 prompt 的偏好判断应一致 |
| 多样性 | 覆盖不同类型的任务和回答风格 |
| 难度 | 包含一些难以区分的边界案例 |
| 数量 | 建议至少 50K 条偏好对比对 |

**数据预处理脚本：**

```python
"""prepare_rm_data.py - 奖励模型数据预处理"""
import json
from nexus_llm import NexusTokenizer

tokenizer = NexusTokenizer.from_pretrained("nexus-ai/nexus-7b")

def process_rm_data(input_file: str, output_file: str, max_length: int = 2048):
    """处理偏好数据为奖励模型训练格式。"""
    processed = []
    skipped = 0

    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prompt = data["prompt"]
            chosen = data["chosen"]
            rejected = data["rejected"]

            # 构建完整文本
            chosen_text = f"### 指令:\n{prompt}\n\n### 回答:\n{chosen}"
            rejected_text = f"### 指令:\n{prompt}\n\n### 回答:\n{rejected}"

            chosen_tokens = tokenizer.encode(chosen_text)
            rejected_tokens = tokenizer.encode(rejected_text)

            if len(chosen_tokens) > max_length or len(rejected_tokens) > max_length:
                skipped += 1
                continue

            processed.append({
                "chosen_input_ids": chosen_tokens,
                "rejected_input_ids": rejected_tokens,
                "chosen_attention_mask": [1] * len(chosen_tokens),
                "rejected_attention_mask": [1] * len(rejected_tokens),
            })

    with open(output_file, 'w') as f:
        for item in processed:
            f.write(json.dumps(item) + '\n')

    print(f"处理完成: {len(processed)} 条有效, {skipped} 条跳过")

if __name__ == "__main__":
    process_rm_data("./data/raw_preferences.jsonl", "./data/rm_processed.jsonl")
```

### 奖励模型配置

```python
"""train_reward_model.py - 奖励模型训练脚本"""
from nexus_llm import NexusForCausalLM, NexusTokenizer, NexusConfig
from nexus_llm.rlhf import RewardModel, RewardConfig, RewardTrainer

# 加载 SFT 模型作为初始化
sft_model = NexusForCausalLM.from_pretrained("./output/sft/merged-model")
tokenizer = NexusTokenizer.from_pretrained("./output/sft/merged-model")

# 奖励模型配置
reward_config = RewardConfig(
    model_name="nexus-7b-reward",
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    dropout=0.1,
)

# 从 SFT 模型初始化奖励模型
reward_model = RewardModel.from_pretrained(
    "./output/sft/merged-model",
    config=reward_config,
)

# 训练配置
training_config = TrainingConfig(
    output_dir="./output/reward_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,          # RM 学习率通常较小
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    save_total_limit=3,
)

# 创建奖励模型训练器
trainer = RewardTrainer(
    model=reward_model,
    config=training_config,
    train_file="./data/rm_processed.jsonl",
    eval_file="./data/rm_eval.jsonl",
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()

# 保存奖励模型
trainer.save_model("./output/reward_model/final")
```

### 训练奖励模型

```bash
# 使用 Python 脚本
python train_reward_model.py

# 使用命令行工具
nexus-train \
  --model-path ./output/sft/merged-model \
  --train-file ./data/rm_processed.jsonl \
  --output-dir ./output/reward_model \
  --task reward_modeling \
  --num-epochs 1 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --fp16 \
  --gradient-checkpointing
```

### 评估奖励模型

```python
"""eval_reward_model.py - 奖励模型评估"""
from nexus_llm.rlhf import RewardModel
import json

# 加载奖励模型
reward_model = RewardModel.from_pretrained("./output/reward_model/final")

# 测试用例
test_cases = [
    {
        "prompt": "解释相对论",
        "good": "相对论是爱因斯坦提出的物理学理论，分为狭义相对论和广义相对论...",
        "bad": "相对论就是关于相对的东西。",
    },
    {
        "prompt": "写一个排序算法",
        "good": "```python\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n```",
        "bad": "排序就是排一下。",
    },
]

# 评估准确率
correct = 0
total = len(test_cases)

for case in test_cases:
    good_text = f"### 指令:\n{case['prompt']}\n\n### 回答:\n{case['good']}"
    bad_text = f"### 指令:\n{case['prompt']}\n\n### 回答:\n{case['bad']}"

    good_score = reward_model(good_text).item()
    bad_score = reward_model(bad_text).item()

    is_correct = good_score > bad_score
    correct += int(is_correct)

    print(f"Prompt: {case['prompt'][:30]}...")
    print(f"  Good: {good_score:.4f}, Bad: {bad_score:.4f}, Correct: {is_correct}")

accuracy = correct / total
print(f"\n准确率: {accuracy:.2%} ({correct}/{total})")
```

---

## 第三步：PPO 训练

PPO（Proximal Policy Optimization）阶段使用奖励模型的反馈信号，通过强化学习优化策略模型，使其生成人类更偏好的回答。

### PPO 原理简介

```
PPO 训练循环:

1. 从提示数据集中采样一批 prompt
2. 使用当前策略模型生成回答 (response)
3. 使用奖励模型对回答打分 (reward)
4. 计算奖励信号（可能包含 KL 惩罚）
5. 计算 PPO 损失并更新策略模型

损失函数:
  L = -E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]

  其中:
    r_t = pi_new(a|s) / pi_old(a|s)  -- 新旧策略概率比
    A_t = Q_t - V_t                    -- 优势函数
    eps = 0.2                          -- 裁剪范围
```

### PPO 配置

```python
"""ppo_train.py - PPO 训练脚本"""
from nexus_llm import NexusForCausalLM, NexusTokenizer
from nexus_llm.rlhf import PPOTrainer, PPOConfig, RewardModel

# 加载模型
policy_model = NexusForCausalLM.from_pretrained("./output/sft/merged-model")
reference_model = NexusForCausalLM.from_pretrained("./output/sft/merged-model")
reward_model = RewardModel.from_pretrained("./output/reward_model/final")
tokenizer = NexusTokenizer.from_pretrained("./output/sft/merged-model")

# 冻结参考模型和奖励模型
for param in reference_model.parameters():
    param.requires_grad = False
for param in reward_model.parameters():
    param.requires_grad = False

# PPO 配置
ppo_config = PPOConfig(
    # 学习率
    learning_rate=1e-6,               # PPO 学习率非常小
    # KL 散度控制
    kl_coef=0.1,                      # KL 惩罚系数
    target_kl=0.1,                    # 目标 KL 散度（早停阈值）
    # PPO 超参数
    gamma=1.0,                        # 折扣因子（对话任务通常为 1.0）
    lam=0.95,                         # GAE lambda
    cliprange=0.2,                    # PPO 裁剪范围
    cliprange_value=0.2,              # 价值函数裁剪范围
    # 训练参数
    ppo_epochs=4,                     # 每次 PPO 更新的 epoch 数
    mini_batch_size=128,              # mini-batch 大小
    gradient_accumulation_steps=1,
    # 生成参数
    max_new_tokens=256,               # 最大生成长度
    temperature=0.7,                  # 生成温度
    top_p=0.9,                        # nucleus 采样
    # 损失权重
    vf_coef=0.1,                      # 价值函数损失权重
    entropy_coef=0.01,                # 熵正则化权重
    # 早停
    early_stopping=True,              # KL 超过阈值时早停
)

# 创建 PPO 训练器
trainer = PPOTrainer(
    config=ppo_config,
    policy_model=policy_model,
    reference_model=reference_model,
    reward_model=reward_model,
    tokenizer=tokenizer,
)

# 加载提示数据
import json
prompts = []
with open("./data/ppo_prompts.jsonl", 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        prompts.append(data["prompt"])

# PPO 训练循环
num_ppo_epochs = 10
for epoch in range(num_ppo_epochs):
    # 每轮随机采样一批 prompt
    import random
    batch_prompts = random.sample(prompts, min(256, len(prompts)))

    # 执行 PPO 更新
    stats = trainer.step(batch_prompts)

    # 打印训练指标
    print(f"\nEpoch {epoch + 1}/{num_ppo_epochs}")
    print(f"  平均奖励:     {stats['reward_mean']:.4f}")
    print(f"  策略损失:     {stats['policy_loss']:.4f}")
    print(f"  价值损失:     {stats['value_loss']:.4f}")
    print(f"  KL 散度:      {stats['kl_divergence']:.4f}")
    print(f"  策略熵:       {stats['entropy']:.4f}")
    print(f"  新/旧策略比:  {stats['approx_kl']:.4f}")

# 保存对齐后的模型
policy_model.save_pretrained("./output/ppo/final-model")
tokenizer.save_pretrained("./output/ppo/final-model")
```

### 运行 PPO 训练

```bash
# 使用 Python 脚本
python ppo_train.py

# 使用命令行工具
nexus-train \
  --model-path ./output/sft/merged-model \
  --reward-model-path ./output/reward_model/final \
  --train-file ./data/ppo_prompts.jsonl \
  --output-dir ./output/ppo \
  --task ppo \
  --ppo-learning-rate 1e-6 \
  --ppo-kl-coef 0.1 \
  --ppo-epochs 4 \
  --ppo-mini-batch-size 128 \
  --ppo-max-new-tokens 256 \
  --ppo-temperature 0.7 \
  --fp16 \
  --num-gpus 4
```

### 监控训练过程

PPO 训练需要密切监控以下关键指标：

```python
"""monitor_ppo.py - PPO 训练监控"""
import matplotlib.pyplot as plt

# 关键指标可视化
metrics = {
    "reward_mean": [],       # 平均奖励（应逐步上升）
    "kl_divergence": [],     # KL 散度（应保持在 target_kl 附近）
    "policy_loss": [],       # 策略损失
    "value_loss": [],        # 价值损失
    "entropy": [],           # 策略熵（下降过快说明模式崩塌）
    "approx_kl": [],         # 近似 KL（用于早停判断）
}

# 健康指标判断
def check_health(stats):
    warnings = []

    # KL 散度过大 - 策略偏离参考模型太多
    if stats['kl_divergence'] > 0.5:
        warnings.append("KL 散度过大！策略可能已经偏离参考模型。")

    # 熵过低 - 模型可能模式崩塌
    if stats['entropy'] < 0.5:
        warnings.append("策略熵过低！模型可能发生模式崩塌。")

    # 奖励为负 - 模型生成质量差
    if stats['reward_mean'] < 0:
        warnings.append("平均奖励为负！模型生成质量可能较差。")

    # 价值损失过大 - 价值函数拟合不好
    if stats['value_loss'] > 10.0:
        warnings.append("价值损失过大！考虑增大 vf_coef 或调整学习率。")

    return warnings
```

**关键指标健康范围：**

| 指标 | 健康范围 | 异常信号 |
|------|---------|---------|
| `reward_mean` | 持续上升 | 下降或剧烈波动 |
| `kl_divergence` | 0.01 ~ 0.2 | > 0.5（策略偏离过大） |
| `entropy` | > 1.0 | < 0.5（模式崩塌） |
| `policy_loss` | 0.01 ~ 1.0 | > 10.0（训练不稳定） |
| `value_loss` | 0.1 ~ 5.0 | > 10.0（价值函数发散） |

---

## 配置参考

### SFTConfig

```python
class SFTConfig(TrainingConfig):
    """SFT 训练配置。

    继承 TrainingConfig 的所有参数，增加以下参数:

    参数:
        template (str): 对话模板。可选: "alpaca", "chatml", "sharegpt"。默认: "alpaca"
        packing (bool): 是否将多个样本打包到一个序列中。默认: True
        max_seq_length (int): 最大序列长度。默认: 2048
        label_masking (bool): 是否对 prompt 部分屏蔽损失。默认: True
    """
```

### RewardModelConfig

```python
class RewardModelConfig(NexusConfig):
    """奖励模型配置。

    继承 NexusConfig 的所有参数，增加以下参数:

    参数:
        num_pooled_outputs (int): 池化输出数量。默认: 1
        reward_head_hidden_size (int): 奖励头隐藏层维度。默认: None
        dropout (float): Dropout 概率。默认: 0.1
        loss_type (str): 损失类型。可选: "ranking", "margin", "mse"。默认: "ranking"
        margin (float): margin loss 的间隔。默认: 0.5
    """
```

### PPOConfig

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
        top_p (float): nucleus 采样阈值。默认: 0.9
        early_stopping (bool): 是否启用早停。默认: True
        target_kl (float): 目标 KL 散度。默认: 0.1
        whiten_rewards (bool): 是否对奖励进行白化处理。默认: True
        reward_clip_range (tuple): 奖励裁剪范围。默认: (-5.0, 5.0)
    """
```

---

## 最佳实践

### SFT 阶段

1. **数据质量优先于数量**：10K 条高质量数据优于 100K 条低质量数据
2. **使用 LoRA 微调**：节省显存，训练速度更快，效果接近全参数微调
3. **适当的学习率**：SFT 推荐 1e-5 ~ 5e-5，过大容易过拟合
4. **监控过拟合**：定期在验证集上评估，loss 上升时提前停止
5. **数据多样性**：确保训练数据覆盖模型需要处理的各种任务类型

### 奖励模型阶段

1. **偏好对比要明确**：chosen 和 rejected 之间应有显著的质量差异
2. **使用 SFT 模型初始化**：从 SFT 模型初始化比从基础模型初始化效果更好
3. **较小的学习率**：奖励模型推荐 5e-6 ~ 2e-5
4. **评估准确率**：确保奖励模型在测试集上的准确率 > 65%
5. **避免奖励黑客**：不要使用过于简单的偏好标准

### PPO 阶段

1. **极小的学习率**：PPO 推荐 1e-7 ~ 5e-6，过大导致训练不稳定
2. **密切监控 KL 散度**：KL 过大说明策略偏离参考模型太远
3. **使用 KL 早停**：当 KL 超过阈值时停止更新，防止策略崩溃
4. **适当的 batch size**：PPO batch size 不宜过小，建议 >= 128
5. **逐步增加 KL 系数**：可以从较小的 kl_coef 开始，逐步增大
6. **奖励白化**：启用 whiten_rewards 可以稳定训练
7. **定期保存检查点**：PPO 训练不稳定，需要保存多个检查点以便回滚

### 资源需求参考

| 模型规模 | SFT (LoRA) | RM 训练 | PPO 训练 |
|---------|-----------|---------|---------|
| 7B | 1x A100 40GB | 1x A100 40GB | 2-4x A100 40GB |
| 13B | 2x A100 40GB | 2x A100 40GB | 4-8x A100 40GB |
| 30B | 4x A100 40GB | 4x A100 40GB | 8x A100 80GB |
| 70B | 4x A100 80GB | 4x A100 80GB | 8x A100 80GB |

---

## 故障排除

### 常见问题

**Q: PPO 训练中奖励持续下降**

```
原因分析:
  1. 学习率过大，策略更新步长太大
  2. KL 惩罚系数太小，策略偏离参考模型过快
  3. 奖励模型质量不足，给出错误的奖励信号

解决方案:
  1. 降低学习率至 1e-7
  2. 增大 kl_coef 至 0.2 ~ 0.5
  3. 检查奖励模型的准确率，考虑重新训练
  4. 启用 whiten_rewards
  5. 减小 cliprange 至 0.1
```

**Q: 模型发生模式崩塌（重复生成相同内容）**

```
原因分析:
  1. 策略熵过低
  2. 生成温度过低
  3. KL 惩罚过强

解决方案:
  1. 增大 entropy_coef 至 0.05
  2. 增大生成温度至 0.9 ~ 1.0
  3. 减小 kl_coef 至 0.05
  4. 增大 top_p 至 0.95
  5. 添加重复惩罚 repetition_penalty=1.1
```

**Q: PPO 训练 OOM（显存不足）**

```
解决方案:
  1. 减小 mini_batch_size
  2. 启用 gradient_checkpointing
  3. 使用 FP16/BF16
  4. 减小 max_new_tokens
  5. 使用 DeepSpeed ZeRO-3
  6. 对策略模型使用 LoRA
```

**Q: 奖励模型准确率过低**

```
原因分析:
  1. 偏好数据质量差
  2. chosen 和 rejected 差异不够明显
  3. 训练不充分

解决方案:
  1. 清洗偏好数据，确保质量
  2. 确保偏好对之间有明确差异
  3. 增加训练轮数
  4. 尝试不同的损失函数（ranking -> margin）
  5. 增大模型规模
```

**Q: SFT 模型在 PPO 后性能退化**

```
原因分析:
  1. PPO 训练过度，偏离 SFT 模型太远
  2. 奖励模型偏向某些特定模式

解决方案:
  1. 增大 KL 惩罚系数
  2. 减少 PPO 训练轮数
  3. 使用 DPO（直接偏好优化）替代 PPO
  4. 在 PPO 后进行少量 SFT 恢复训练
  5. 使用混合目标：L = L_ppo + alpha * L_sft
```

---

> 更多信息请参阅 [API 参考](./api_reference.md) 和 [快速入门](./getting_started.md)。
