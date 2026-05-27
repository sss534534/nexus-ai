# Nexus-LLM 贡献指南

> 感谢你对 Nexus-LLM 项目的关注！本文档将指导你如何参与项目开发，包括环境搭建、代码规范、测试要求和 PR 提交流程。

---

## 目录

- [行为准则](#行为准则)
- [开发环境搭建](#开发环境搭建)
  - [Fork 与克隆](#fork-与克隆)
  - [Python 环境](#python-环境)
  - [安装开发依赖](#安装开发依赖)
  - [IDE 配置](#ide-配置)
  - [验证环境](#验证环境)
- [代码规范](#代码规范)
  - [代码风格 (Black + isort)](#代码风格-black--isort)
  - [代码检查 (Flake8)](#代码检查-flake8)
  - [类型注解 (mypy)](#类型注解-mypy)
  - [文档字符串 (Google Style)](#文档字符串-google-style)
  - [命名规范](#命名规范)
  - [导入规范](#导入规范)
  - [Git 提交规范](#git-提交规范)
- [测试指南](#测试指南)
  - [测试结构](#测试结构)
  - [编写单元测试](#编写单元测试)
  - [编写集成测试](#编写集成测试)
  - [运行测试](#运行测试)
  - [测试覆盖率](#测试覆盖率)
  - [Mock 与 Fixture](#mock-与-fixture)
- [PR 提交流程](#pr-提交流程)
  - [分支管理](#分支管理)
  - [提交 PR](#提交-pr)
  - [PR 模板](#pr-模板)
  - [代码审查](#代码审查)
- [Issue 报告](#issue-报告)
  - [Bug 报告](#bug-报告)
  - [功能请求](#功能请求)
  - [文档问题](#文档问题)

---

## 行为准则

Nexus-LLM 社区遵循以下原则：

- **尊重他人**：对所有贡献者保持尊重和友善
- **包容开放**：欢迎不同背景和经验的开发者参与
- **建设性反馈**：以建设性的方式提出意见和建议
- **专注技术**：讨论围绕技术问题展开，避免无关话题

---

## 开发环境搭建

### Fork 与克隆

```bash
# 1. 在 GitHub 上 Fork 项目
# 访问 https://github.com/nexus-ai/nexus-llm，点击 Fork 按钮

# 2. 克隆你的 Fork
git clone https://github.com/<your-username>/nexus-llm.git
cd nexus-llm

# 3. 添加上游仓库（保持同步）
git remote add upstream https://github.com/nexus-ai/nexus-llm.git

# 4. 验证远程仓库
git remote -v
# origin    https://github.com/<your-username>/nexus-llm.git (fetch)
# origin    https://github.com/<your-username>/nexus-llm.git (push)
# upstream  https://github.com/nexus-ai/nexus-llm.git (fetch)
# upstream  https://github.com/nexus-ai/nexus-llm.git (push)
```

### Python 环境

```bash
# 1. 安装 Python 3.10+（推荐使用 pyenv）
# macOS
brew install pyenv
pyenv install 3.11.5
pyenv global 3.11.5

# Ubuntu
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# 2. 创建虚拟环境
python -m venv nexus-dev
source nexus-dev/bin/activate

# 3. 升级 pip
pip install --upgrade pip setuptools wheel
```

### 安装开发依赖

```bash
# 安装项目（可编辑模式 + 开发依赖）
pip install -e ".[dev]"

# 开发依赖包含:
# - black: 代码格式化
# - isort: 导入排序
# - flake8: 代码检查
# - mypy: 类型检查
# - pytest: 测试框架
# - pytest-cov: 测试覆盖率
# - pre-commit: Git 预提交钩子
# - sphinx: 文档生成
```

### IDE 配置

**VS Code 推荐：**

创建 `.vscode/settings.json`：

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/nexus-dev/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.sortImports.provider": "isort",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": "explicit"
  },
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "[python]": {
    "editor.tabSize": 4,
    "editor.rulers": [88]
  }
}
```

**PyCharm 推荐：**

- Settings -> Tools -> Black -> 勾选 "Use Black to format code"
- Settings -> Tools -> External Tools -> 添加 isort
- Settings -> Editor -> Code Style -> Python -> 设置 Tab size 为 4

### 验证环境

```bash
# 验证 Python 版本
python --version  # 应为 3.9+

# 验证安装
python -c "import nexus_llm; print(nexus_llm.__version__)"

# 验证 CUDA（如需 GPU 支持）
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 运行代码格式化检查
black --check nexus_llm/
isort --check-only nexus_llm/

# 运行代码检查
flake8 nexus_llm/

# 运行测试
pytest tests/ -v
```

---

## 代码规范

### 代码风格 (Black + isort)

Nexus-LLM 使用 **Black** 作为代码格式化工具，**isort** 作为导入排序工具。

**Black 配置 (`pyproject.toml`)：**

```toml
[tool.black]
line-length = 88
target-version = ["py39", "py310", "py311"]
include = '\.pyi?$'
extend-exclude = '''
/(
    \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''
```

**isort 配置 (`pyproject.toml`)：**

```toml
[tool.isort]
profile = "black"
line_length = 88
known_first_party = ["nexus_llm"]
known_third_party = ["torch", "transformers", "fastapi", "pydantic"]
force_single_line = false
lines_after_imports = 2
```

**使用方法：**

```bash
# 格式化代码
black nexus_llm/ tests/
isort nexus_llm/ tests/

# 检查格式（不修改文件）
black --check nexus_llm/ tests/
isort --check-only nexus_llm/ tests/

# 自动修复
black --diff nexus_llm/ tests/  # 查看差异
```

**格式化示例：**

```python
# 格式化前
from nexus_llm import NexusConfig,NexusForCausalLM,NexusTokenizer
from nexus_llm.training import Trainer
import torch,torch.nn as nn
import numpy as np
def my_function(  x,y,z  ):
    result=x+y+z
    return result

# 格式化后
import torch
import torch.nn as nn

import numpy as np

from nexus_llm import NexusConfig, NexusForCausalLM, NexusTokenizer
from nexus_llm.training import Trainer


def my_function(x, y, z):
    result = x + y + z
    return result
```

### 代码检查 (Flake8)

```toml
[tool.flake8]
max-line-length = 88
extend-ignore = ["E203", "E501", "W503"]
exclude = [
    ".git",
    "__pycache__",
    "build",
    "dist",
    "*.egg-info",
]
per-file-ignores = [
    "__init__.py:F401",
    "tests/*:S101",
]
```

**使用方法：**

```bash
# 运行 Flake8 检查
flake8 nexus_llm/

# 指定文件
flake8 nexus_llm/training/trainer.py

# 显示统计
flake8 nexus_llm/ --statistics --count
```

**常见错误及修复：**

| 错误码 | 说明 | 修复方法 |
|--------|------|---------|
| E501 | 行过长 | 拆分为多行或使用括号续行 |
| F401 | 未使用的导入 | 删除导入或添加 `# noqa: F401` |
| F841 | 未使用的变量 | 删除变量或使用 `_` 前缀 |
| E203 | 列表冒号前空格 | Black 会自动处理 |
| W291 | 行尾空格 | 删除行尾空格 |
| E302 | 缺少空行 | 函数/类之间添加两个空行 |

### 类型注解 (mypy)

```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
strict_equality = true

[[tool.mypy.overrides]]
module = ["torch.*", "transformers.*", "vllm.*"]
ignore_missing_imports = true
```

**使用方法：**

```bash
# 运行类型检查
mypy nexus_llm/

# 指定文件
mypy nexus_llm/training/trainer.py

# 严格模式
mypy nexus_llm/ --strict
```

**类型注解示例：**

```python
from typing import Optional, List, Dict, Tuple, Union
import torch
from torch import Tensor


def compute_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor] = None,
    dropout: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """计算注意力权重和输出。

    Args:
        query: 查询张量，形状为 (batch, heads, seq_len, head_dim)。
        key: 键张量，形状为 (batch, heads, seq_len, head_dim)。
        value: 值张量，形状为 (batch, heads, seq_len, head_dim)。
        attention_mask: 可选的注意力掩码。
        dropout: Dropout 概率。

    Returns:
        Tuple[Tensor, Tensor]: 注意力输出和注意力权重。
    """
    ...
```

### 文档字符串 (Google Style)

Nexus-LLM 使用 Google 风格的文档字符串：

```python
class NexusModel(nn.Module):
    """Nexus-LLM 基础模型。

    包含嵌入层、多层 Transformer 和最终的 LayerNorm。
    不包含语言模型头（LM Head）。

    Attributes:
        config: 模型配置实例。
        embed_tokens: 词嵌入层。
        layers: Transformer 层列表。
        norm: 最终的 RMSNorm 层。

    Examples:
        >>> config = NexusConfig(hidden_size=4096, num_hidden_layers=32)
        >>> model = NexusModel(config)
        >>> outputs = model(input_ids=tokenizer("hello", return_tensors="pt").input_ids)
        >>> print(outputs.last_hidden_state.shape)
        torch.Size([1, 5, 4096])
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
        """模型前向传播。

        Args:
            input_ids: 输入 token ID，形状为 (batch_size, seq_length)。
            attention_mask: 注意力掩码，形状为 (batch_size, seq_length)。
                默认为 None，表示不使用掩码。
            position_ids: 位置 ID，形状为 (batch_size, seq_length)。
                默认为 None，自动生成。
            past_key_values: 过去的 KV Cache，用于增量生成。
            inputs_embeds: 直接传入的嵌入表示（与 input_ids 二选一）。
            use_cache: 是否返回 KV Cache。默认为 None，使用 config 中的设置。

        Returns:
            BaseModelOutputWithPast: 包含以下字段:
                - last_hidden_state (Tensor): 最后一层隐藏状态
                - past_key_values (List): 更新后的 KV Cache
                - hidden_states (Optional[List[Tensor]]): 所有层的隐藏状态
                - attentions (Optional[List[Tensor]]): 所有层的注意力权重

        Raises:
            ValueError: 当同时提供 input_ids 和 inputs_embeds 时。
            RuntimeError: 当输入张量维度不匹配时。
        """
        ...
```

### 命名规范

```python
# 模块和包: 小写，下划线分隔
nexus_llm/
    core/
        attention.py
        model.py
    training/
        distributed_trainer.py

# 类名: 大驼峰 (PascalCase)
class NexusConfig: ...
class DistributedTrainer: ...
class KnowledgeDistiller: ...

# 函数和变量: 小写，下划线分隔 (snake_case)
def compute_loss(predictions, targets): ...
max_sequence_length = 4096
attention_mask = ...

# 常量: 大写，下划线分隔
MAX_POSITION_EMBEDDINGS = 8192
DEFAULT_LEARNING_RATE = 2e-5
RMS_NORM_EPS = 1e-6

# 私有方法/变量: 单下划线前缀
def _compute_attention_scores(self): ...
_internal_state = {}

# 特殊方法: 双下划线前缀（谨慎使用）
class Base:
    def __init__(self): ...
    def __repr__(self): ...

# 布尔变量: 使用 is/has 前缀
is_training = True
has_cache = False
use_fp16 = True

# 避免的命名
# 不好: l (小写L), O (大写O), I (大写I) - 容易与数字混淆
# 不好: func1, func2, tmp - 缺乏语义
# 不好: data, info, result - 过于笼统
```

### 导入规范

```python
# 1. 标准库
import os
import sys
import json
from typing import Optional, List, Dict, Tuple
from pathlib import Path

# 2. 第三方库
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 3. 本地模块
from nexus_llm import NexusConfig, NexusForCausalLM
from nexus_llm.core import attention
from nexus_llm.training import Trainer, TrainingConfig
from nexus_llm.serving import InferenceEngine

# 避免通配符导入
# 不好: from nexus_llm.core import *
# 好: from nexus_llm.core import NexusConfig, NexusModel

# 避免循环导入
# 如果模块 A 需要模块 B 的类型，使用 TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexus_llm.training import Trainer
```

### Git 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**类型 (type)：**

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修复 Bug |
| `docs` | 文档变更 |
| `style` | 代码格式（不影响功能） |
| `refactor` | 重构（非新功能、非修复） |
| `perf` | 性能优化 |
| `test` | 测试相关 |
| `chore` | 构建/工具/依赖变更 |
| `ci` | CI/CD 配置变更 |

**示例：**

```bash
# 新功能
git commit -m "feat(training): add LoRA fine-tuning support"

# 修复 Bug
git commit -m "fix(serving): resolve memory leak in streaming response"

# 文档
git commit -m "docs(api): update InferenceEngine API reference"

# 性能优化
git commit -m "perf(attention): optimize Flash Attention kernel for GQA"

# 重构
git commit -m "refactor(core): extract RMSNorm into separate module"

# 测试
git commit -m "test(training): add unit tests for DistributedTrainer"
```

---

## 测试指南

### 测试结构

```
tests/
|
|-- conftest.py                 # 全局 Fixture
|
|-- unit/                       # 单元测试
|   |-- __init__.py
|   |-- test_config.py          # 配置类测试
|   |-- test_model.py           # 模型测试
|   |-- test_tokenizer.py       # 分词器测试
|   |-- test_attention.py       # 注意力机制测试
|   |-- test_trainer.py         # 训练器测试
|   |-- test_engine.py          # 推理引擎测试
|   |-- test_rlhf.py            # RLHF 模块测试
|   `-- test_agent.py           # Agent 模块测试
|
|-- integration/                # 集成测试
|   |-- __init__.py
|   |-- test_training_pipeline.py
|   |-- test_serving_pipeline.py
|   `-- test_rlhf_pipeline.py
|
|-- benchmarks/                 # 性能基准测试
|   |-- __init__.py
|   |-- bench_inference.py
|   `-- bench_training.py
|
`-- fixtures/                   # 测试数据
    |-- sample_config.json
    |-- sample_data.jsonl
    `-- sample_model/
```

### 编写单元测试

```python
"""tests/unit/test_model.py - 模型单元测试"""
import pytest
import torch
from nexus_llm import NexusConfig, NexusModel, NexusForCausalLM


class TestNexusConfig:
    """NexusConfig 测试类。"""

    def test_default_config(self):
        """测试默认配置创建。"""
        config = NexusConfig()
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 32
        assert config.max_position_embeddings == 4096

    def test_custom_config(self):
        """测试自定义配置。"""
        config = NexusConfig(
            hidden_size=5120,
            num_hidden_layers=48,
            num_attention_heads=40,
        )
        assert config.hidden_size == 5120
        assert config.num_hidden_layers == 48
        assert config.num_attention_heads == 40

    def test_config_serialization(self):
        """测试配置序列化和反序列化。"""
        config = NexusConfig(hidden_size=5120)
        config_dict = config.to_dict()
        restored = NexusConfig.from_dict(config_dict)
        assert restored.hidden_size == config.hidden_size

    def test_gqa_config(self):
        """测试 GQA 配置。"""
        config = NexusConfig(
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        assert config.num_key_value_heads == 8
        assert config.num_attention_heads % config.num_key_value_heads == 0


class TestNexusModel:
    """NexusModel 测试类。"""

    @pytest.fixture
    def small_config(self):
        """创建小型测试配置。"""
        return NexusConfig(
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            max_position_embeddings=128,
            vocab_size=1000,
        )

    @pytest.fixture
    def model(self, small_config):
        """创建测试模型。"""
        return NexusModel(small_config)

    def test_forward_pass(self, model, small_config):
        """测试前向传播。"""
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        outputs = model(input_ids)

        assert outputs.last_hidden_state.shape == (batch_size, seq_len, small_config.hidden_size)

    def test_forward_with_attention_mask(self, model, small_config):
        """测试带注意力掩码的前向传播。"""
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 8:] = 0  # 第一个样本只看前 8 个 token

        outputs = model(input_ids, attention_mask=attention_mask)

        assert outputs.last_hidden_state.shape == (batch_size, seq_len, small_config.hidden_size)

    def test_forward_with_cache(self, model, small_config):
        """测试带 KV Cache 的前向传播。"""
        input_ids = torch.randint(0, small_config.vocab_size, (1, 8))

        outputs = model(input_ids, use_cache=True)
        assert outputs.past_key_values is not None

        # 增量生成
        next_ids = torch.randint(0, small_config.vocab_size, (1, 1))
        outputs = model(
            next_ids,
            past_key_values=outputs.past_key_values,
            use_cache=True,
        )
        assert outputs.last_hidden_state.shape == (1, 1, small_config.hidden_size)

    def test_gradient_flow(self, model, small_config):
        """测试梯度流。"""
        input_ids = torch.randint(0, small_config.vocab_size, (1, 8))
        outputs = model(input_ids)
        loss = outputs.last_hidden_state.sum()
        loss.backward()

        # 检查所有参数都有梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 {name} 没有梯度"


class TestNexusForCausalLM:
    """NexusForCausalLM 测试类。"""

    @pytest.fixture
    def small_config(self):
        return NexusConfig(
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
            vocab_size=1000,
        )

    def test_loss_computation(self, small_config):
        """测试损失计算。"""
        model = NexusForCausalLM(small_config)
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)

        assert outputs.loss is not None
        assert outputs.loss.requires_grad
        assert outputs.logits.shape == (2, 16, 1000)

    def test_generate(self, small_config):
        """测试文本生成。"""
        model = NexusForCausalLM(small_config)
        input_ids = torch.randint(0, 1000, (1, 4))

        output_ids = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,  # 贪心解码，结果确定
        )

        assert output_ids.shape[0] == 1
        assert output_ids.shape[1] > 4  # 生成了新 token
```

### 编写集成测试

```python
"""tests/integration/test_training_pipeline.py - 训练流水线集成测试"""
import pytest
import torch
from nexus_llm import NexusConfig, NexusForCausalLM
from nexus_llm.training import Trainer, TrainingConfig


class TestTrainingPipeline:
    """训练流水线集成测试。"""

    @pytest.fixture
    def tiny_model(self):
        """创建微型模型用于快速测试。"""
        config = NexusConfig(
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=256,
            max_position_embeddings=64,
            vocab_size=500,
        )
        return NexusForCausalLM(config)

    @pytest.fixture
    def sample_data(self, tmp_path):
        """创建测试数据。"""
        import json
        data_file = tmp_path / "train.jsonl"
        with open(data_file, "w") as f:
            for i in range(20):
                f.write(json.dumps({
                    "input_ids": [1, 2, 3, 4, 5],
                    "labels": [2, 3, 4, 5, 6],
                    "attention_mask": [1, 1, 1, 1, 1],
                }) + "\n")
        return str(data_file)

    def test_basic_training(self, tiny_model, sample_data, tmp_path):
        """测试基本训练流程。"""
        config = TrainingConfig(
            output_dir=str(tmp_path / "output"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            max_steps=5,  # 仅训练 5 步用于快速测试
            logging_steps=1,
            save_steps=5,
        )

        trainer = Trainer(
            model=tiny_model,
            config=config,
            train_file=sample_data,
        )

        result = trainer.train()

        assert result.global_step == 5
        assert result.training_loss is not None

    def test_model_save_and_load(self, tiny_model, sample_data, tmp_path):
        """测试模型保存和加载。"""
        output_dir = str(tmp_path / "output")

        config = TrainingConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            max_steps=3,
        )

        trainer = Trainer(
            model=tiny_model,
            config=config,
            train_file=sample_data,
        )
        trainer.train()
        trainer.save_model()

        # 验证文件存在
        import os
        assert os.path.exists(os.path.join(output_dir, "config.json"))
        assert os.path.exists(os.path.join(output_dir, "model.safetensors"))

        # 加载模型
        loaded_model = NexusForCausalLM.from_pretrained(output_dir)
        assert loaded_model is not None
```

### 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 运行指定文件
pytest tests/unit/test_model.py -v

# 运行指定测试类
pytest tests/unit/test_model.py::TestNexusModel -v

# 运行指定测试方法
pytest tests/unit/test_model.py::TestNexusModel::test_forward_pass -v

# 并行运行（需要 pytest-xdist）
pytest tests/ -v -n auto

# 仅运行上次失败的测试
pytest tests/ --lf

# 显示详细输出
pytest tests/ -v -s

# 按名称过滤
pytest tests/ -k "test_attention" -v

# 跳过慢速测试
pytest tests/ -v -m "not slow"

# 仅运行标记为 slow 的测试
pytest tests/ -v -m "slow"
```

### 测试覆盖率

```bash
# 生成覆盖率报告
pytest tests/ --cov=nexus_llm --cov-report=html --cov-report=term-missing

# 查看覆盖率阈值
pytest tests/ --cov=nexus_llm --cov-fail-under=80

# 仅查看特定模块的覆盖率
pytest tests/unit/test_model.py --cov=nexus_llm.core.model --cov-report=term-missing
```

**覆盖率要求：**

| 模块 | 最低覆盖率 | 目标覆盖率 |
|------|-----------|-----------|
| 核心模型 (`core/`) | 85% | 95% |
| 训练 (`training/`) | 80% | 90% |
| 推理服务 (`serving/`) | 75% | 90% |
| RLHF (`rlhf/`) | 75% | 85% |
| Agent (`agent/`) | 70% | 85% |
| 压缩 (`compression/`) | 70% | 85% |

### Mock 与 Fixture

```python
"""tests/conftest.py - 全局 Fixture"""
import pytest
import torch
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_config():
    """创建测试用的小型配置。"""
    from nexus_llm import NexusConfig
    return NexusConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
        vocab_size=1000,
    )


@pytest.fixture
def sample_model(sample_config):
    """创建测试用的小型模型。"""
    from nexus_llm import NexusForCausalLM
    model = NexusForCausalLM(sample_config)
    model.eval()  # 设为评估模式
    return model


@pytest.fixture
def mock_tokenizer():
    """创建 Mock 分词器。"""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "hello world"
    tokenizer.vocab_size = 1000
    return tokenizer


@pytest.fixture
def tmp_data_file(tmp_path):
    """创建临时测试数据文件。"""
    import json
    data_file = tmp_path / "test_data.jsonl"
    with open(data_file, "w") as f:
        for i in range(10):
            f.write(json.dumps({"text": f"sample text {i}"}) + "\n")
    return str(data_file)


# 使用 Mock 的示例
def test_with_mock():
    """使用 Mock 测试外部依赖。"""
    with patch("nexus_llm.serving.engine.load_model") as mock_load:
        mock_load.return_value = MagicMock()
        # ... 测试代码
        mock_load.assert_called_once()
```

---

## PR 提交流程

### 分支管理

```bash
# 1. 同步上游代码
git checkout main
git pull upstream main

# 2. 创建功能分支（从 main 分出）
git checkout -b feat/your-feature-name

# 分支命名规范:
# feat/xxx    - 新功能
# fix/xxx     - Bug 修复
# docs/xxx    - 文档更新
# refactor/xxx - 重构
# perf/xxx    - 性能优化
# test/xxx    - 测试相关

# 3. 开发过程中定期同步上游
git fetch upstream
git rebase upstream/main

# 4. 推送到你的 Fork
git push origin feat/your-feature-name
```

### 提交 PR

```bash
# 1. 确保代码通过所有检查
# 格式化
black nexus_llm/ tests/
isort nexus_llm/ tests/

# 代码检查
flake8 nexus_llm/
mypy nexus_llm/

# 测试
pytest tests/ -v
pytest tests/ --cov=nexus_llm --cov-fail-under=80

# 2. 推送最终代码
git push origin feat/your-feature-name

# 3. 在 GitHub 上创建 PR
# 访问 https://github.com/nexus-ai/nexus-llm/compare
# 选择你的分支，填写 PR 描述
```

### PR 模板

```markdown
## 概述

简要描述此 PR 的目的和变更内容。

## 变更类型

请勾选适用的类型:
- [ ] 新功能 (feat)
- [ ] Bug 修复 (fix)
- [ ] 文档更新 (docs)
- [ ] 代码重构 (refactor)
- [ ] 性能优化 (perf)
- [ ] 测试 (test)
- [ ] 其他

## 变更详情

### 主要变更
1. ...
2. ...
3. ...

### 修改的文件
- `nexus_llm/core/attention.py` - 添加了 GQA 支持
- `tests/unit/test_attention.py` - 添加了 GQA 测试
- `docs/api_reference.md` - 更新了 API 文档

## 测试

- [ ] 所有现有测试通过
- [ ] 添加了新的单元测试
- [ ] 添加了新的集成测试
- [ ] 测试覆盖率达到要求

## 检查清单

- [ ] 代码通过 Black 格式化
- [ ] 代码通过 isort 排序
- [ ] 代码通过 Flake8 检查
- [ ] 代码通过 mypy 类型检查
- [ ] 添加了必要的文档字符串
- [ ] 更新了相关文档
- [ ] 遵循了 Git 提交规范

## 关联 Issue

Closes #123
Related to #456

## 备注

（可选）其他需要审查者注意的信息。
```

### 代码审查

**审查者关注点：**

1. **正确性**：代码逻辑是否正确，边界情况是否处理
2. **性能**：是否有明显的性能问题或不必要的计算
3. **安全性**：是否存在安全隐患（SQL 注入、路径遍历等）
4. **可读性**：代码是否清晰易懂，命名是否恰当
5. **测试**：测试是否充分覆盖了变更
6. **文档**：文档字符串和注释是否完整准确
7. **兼容性**：变更是否向后兼容

**审查流程：**

```
提交 PR -> CI 自动检查 -> 代码审查 -> 修改（如需）-> 审查通过 -> 合并
```

---

## Issue 报告

### Bug 报告

提交 Bug 报告时，请使用以下模板：

```markdown
## Bug 描述

简要描述遇到的问题。

## 复现步骤

1. 安装版本：`pip install nexus-llm==x.x.x`
2. 运行以下代码：
   ```python
   # 复现代码
   ```
3. 观察到错误：`错误信息`

## 期望行为

描述你期望的正确行为。

## 实际行为

描述实际发生的情况。

## 环境信息

- OS: Ubuntu 22.04
- Python: 3.11.5
- PyTorch: 2.1.0
- CUDA: 12.1
- Nexus-LLM: 0.1.0
- GPU: NVIDIA A100 40GB

## 错误日志

```
完整的错误堆栈信息
```

## 附加信息

（可选）截图、配置文件或其他有助于诊断问题的信息。
```

### 功能请求

```markdown
## 功能描述

描述你希望添加的功能。

## 动机

为什么需要这个功能？它解决了什么问题？

## 建议的实现方式

（可选）你对实现方式的建议。

## 替代方案

（可选）你考虑过的其他解决方案。

## 附加信息

（可选）参考资料、示例代码等。
```

### 文档问题

```markdown
## 问题描述

描述文档中存在的问题（错误、遗漏、不清楚等）。

## 位置

- 文件：`docs/xxx.md`
- 章节：xxx

## 建议修改

（可选）你对文档修改的建议。
```

---

> 感谢你为 Nexus-LLM 项目做出贡献！如有任何问题，欢迎在 GitHub Issues 中讨论。
