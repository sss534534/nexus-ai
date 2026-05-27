"""
Incremental Fine-tuning Module - 增量微调模块
支持持续学习、增量训练，防止灾难性遗忘
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import numpy as np
from collections import defaultdict

from . import NexusConfig, NexusForCausalLM
from ..training import Trainer, TrainingConfig


logger = logging.getLogger(__name__)


@dataclass
class IncrementalConfig:
    """增量微调配置"""
    
    # 训练策略
    strategy: str = "adapter"  # adapter, replay, regularization, elastic
    
    # Adapter配置
    adapter_hidden_dim: int = 128
    adapter_alpha: float = 0.5
    
    # Replay配置
    replay_buffer_size: int = 10000
    replay_ratio: float = 0.3  # replay样本占总batch的比例
    
    # 正则化配置
    ewc_lambda: float = 5000  # EWC正则化系数
    ewc_fisher_samples: int = 100
    
    # 知识蒸馏配置
    distill_alpha: float = 0.5
    distill_temperature: float = 2.0
    
    # 混合训练
    mix_ratio: float = 0.5  # 新数据与旧数据的混合比例
    
    # 评估
    eval_on_old_tasks: bool = True
    eval_on_new_tasks: bool = True


class ExperienceReplayBuffer:
    """
    经验回放缓冲区
    存储历史任务样本，用于持续学习
    """
    
    def __init__(self, max_size: int = 10000, sample_strategy: str = "uniform"):
        self.max_size = max_size
        self.sample_strategy = sample_strategy
        
        self.buffer: List[Dict[str, Any]] = []
        self.task_counts: Dict[str, int] = defaultdict(int)
        self.priorities: Dict[int, float] = {}  # TD-error based priorities
    
    def add(self, samples: List[Dict[str, Any]], task_name: str = "default"):
        """添加样本到缓冲区"""
        for sample in samples:
            sample["task_name"] = task_name
            self.task_counts[task_name] += 1
            
            # 如果缓冲区满，移除最旧的样本
            if len(self.buffer) >= self.max_size:
                if self.sample_strategy == "uniform":
                    idx_to_remove = 0
                elif self.sample_strategy == "priority":
                    idx_to_remove = min(self.priorities.keys(), 
                                      key=lambda k: self.priorities[k])
                else:
                    idx_to_remove = 0
                
                removed_sample = self.buffer.pop(idx_to_remove)
                self.task_counts[removed_sample["task_name"]] -= 1
                if idx_to_remove in self.priorities:
                    del self.priorities[idx_to_remove]
            
            self.buffer.append(sample)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """从缓冲区采样"""
        if len(self.buffer) == 0:
            return []
        
        if self.sample_strategy == "uniform":
            indices = np.random.choice(
                len(self.buffer),
                size=min(batch_size, len(self.buffer)),
                replace=False,
            ).tolist()
        elif self.sample_strategy == "priority":
            # 基于优先级的采样
            probs = np.array([self.priorities.get(i, 1.0) for i in range(len(self.buffer))])
            probs = probs / probs.sum()
            indices = np.random.choice(
                len(self.buffer),
                size=min(batch_size, len(self.buffer)),
                replace=False,
                p=probs,
            ).tolist()
        elif self.sample_strategy == "task_balanced":
            # 任务平衡采样
            indices = []
            tasks = list(self.task_counts.keys())
            per_task = batch_size // len(tasks)
            
            for task in tasks:
                task_samples = [i for i, s in enumerate(self.buffer) if s["task_name"] == task]
                indices.extend(np.random.choice(task_samples, size=min(per_task, len(task_samples)), replace=False).tolist())
        else:
            indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False).tolist()
        
        return [self.buffer[i] for i in indices]
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """更新样本优先级"""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6
    
    def save(self, path: str):
        """保存缓冲区"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump({
                "buffer": self.buffer,
                "task_counts": dict(self.task_counts),
                "priorities": {str(k): v for k, v in self.priorities.items()},
                "max_size": self.max_size,
                "sample_strategy": self.sample_strategy,
            }, f)
        
        logger.info(f"Replay buffer saved to {path}")
    
    def load(self, path: str):
        """加载缓冲区"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.buffer = data["buffer"]
        self.task_counts = defaultdict(int, data["task_counts"])
        self.priorities = {int(k): v for k, v in data["priorities"].items()}
        self.max_size = data["max_size"]
        self.sample_strategy = data["sample_strategy"]
        
        logger.info(f"Replay buffer loaded from {path}")


class EWCRegularizer:
    """
    Elastic Weight Consolidation (EWC) 正则化器
    通过惩罚对重要参数的大幅度修改来防止灾难性遗忘
    """
    
    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 5000,
    ):
        self.model = model
        self.lambda_ewc = lambda_ewc
        
        # 存储每个任务的最优参数和Fisher信息矩阵
        self.task_params: Dict[str, Dict[str, torch.Tensor]] = {}
        self.task_fisher: Dict[str, Dict[str, torch.Tensor]] = {}
    
    def compute_fisher_information(
        self,
        dataloader,
        num_samples: int = 100,
        device: str = "cuda",
    ) -> Dict[str, torch.Tensor]:
        """
        计算Fisher信息矩阵
        Fisher信息表示参数的重要性
        """
        # 初始化Fisher矩阵
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)
        
        # 收集梯度信息
        self.model.eval()
        samples_collected = 0
        
        for batch in dataloader:
            if samples_collected >= num_samples:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels", input_ids).to(device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs[0] if isinstance(outputs, tuple) else outputs["loss"]
            
            # 反向传播（不更新参数）
            self.model.zero_grad()
            loss.backward()
            
            # 累计梯度平方
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            
            samples_collected += input_ids.size(0)
        
        # 平均
        for name in fisher:
            fisher[name] /= samples_collected
        
        return fisher
    
    def register_task(
        self,
        task_name: str,
        dataloader,
        num_samples: int = 100,
        device: str = "cuda",
    ):
        """注册新任务，记录最优参数和Fisher信息"""
        logger.info(f"Registering task: {task_name}")
        
        # 保存当前参数
        params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.task_params[task_name] = params
        
        # 计算Fisher信息
        fisher = self.compute_fisher_information(dataloader, num_samples, device)
        self.task_fisher[task_name] = fisher
    
    def compute_ewc_loss(self, task_name: str = None) -> torch.Tensor:
        """
        计算EWC损失
        如果指定task_name，只对该任务计算正则化损失
        否则对所有已注册任务计算
        """
        tasks_to_regularize = (
            [task_name] if task_name else list(self.task_params.keys())
        )
        
        ewc_loss = 0.0
        
        for task in tasks_to_regularize:
            if task not in self.task_params:
                continue
            
            for name, param in self.model.named_parameters():
                if name not in self.task_params[task]:
                    continue
                
                # 参数偏离量
                param_diff = param - self.task_params[task][name]
                
                # Fisher加权
                fisher = self.task_fisher[task][name]
                
                # EWC损失 = sum(F * (param - param_opt)^2)
                ewc_loss += (fisher * param_diff ** 2).sum()
        
        return self.lambda_ewc * ewc_loss


class AdapterModule(nn.Module):
    """
    Adapter模块 - 轻量级增量学习组件
    在Transformer层之间插入少量可学习参数
    """
    
    def __init__(
        self,
        hidden_size: int,
        adapter_dim: int = 64,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_dim = adapter_dim
        self.alpha = alpha
        
        # Down-project
        self.down_project = nn.Linear(hidden_size, adapter_dim)
        
        # Up-project
        self.up_project = nn.Linear(adapter_dim, hidden_size)
        
        # 初始化
        nn.init.xavier_uniform_(self.down_project.weight)
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adapter前向传播"""
        # Down-project
        h = self.down_project(x)
        h = F.relu(h)
        
        # Up-project
        h = self.up_project(h)
        
        # 残差连接（带缩放）
        return x + self.alpha * h


class AdapterModel(nn.Module):
    """
    带Adapter的增量学习模型
    冻结原始模型参数，只训练Adapter参数
    """
    
    def __init__(
        self,
        base_model: NexusForCausalLM,
        adapter_dim: int = 64,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.base_model = base_model
        self.adapter_dim = adapter_dim
        self.alpha = alpha
        
        # 冻结基础模型
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 为每层添加Adapter
        self.adapters = nn.ModuleList([
            AdapterModule(
                hidden_size=self.base_model.config.hidden_size,
                adapter_dim=adapter_dim,
                alpha=alpha,
            )
            for _ in range(self.base_model.config.num_hidden_layers)
        ])
        
        # 可训练参数统计
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.base_model.parameters())
        
        logger.info(f"AdapterModel: {trainable_params:,} trainable params ({trainable_params/total_params:.2%})")
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.base_model(*args, **kwargs)
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """获取可训练参数（Adapter参数）"""
        return list(self.adapters.parameters())
    
    def save_adapters(self, path: str):
        """只保存Adapter参数"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(
            self.adapters.state_dict(),
            save_path / "adapters.pt"
        )
        
        logger.info(f"Adapters saved to {path}")
    
    def load_adapters(self, path: str):
        """加载Adapter参数"""
        self.adapters.load_state_dict(
            torch.load(Path(path) / "adapters.pt")
        )
        logger.info(f"Adapters loaded from {path}")


class IncrementalTrainer:
    """
    增量训练器
    支持多种增量学习策略
    """
    
    def __init__(
        self,
        model: NexusForCausalLM,
        tokenizer: Any,
        config: IncrementalConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # 根据策略选择组件
        if config.strategy == "replay":
            self.replay_buffer = ExperienceReplayBuffer(
                max_size=config.replay_buffer_size,
            )
            self.ewc = None
            self.adapters = None
        elif config.strategy == "regularization":
            self.replay_buffer = None
            self.ewc = EWCRegularizer(model, config.ewc_lambda)
            self.adapters = None
        elif config.strategy == "adapter":
            self.replay_buffer = None
            self.ewc = None
            self.adapters = AdapterModel(model, config.adapter_hidden_dim, config.adapter_alpha)
        elif config.strategy == "elastic":
            self.replay_buffer = ExperienceReplayBuffer(max_size=config.replay_buffer_size)
            self.ewc = EWCRegularizer(model, config.ewc_lambda)
            self.adapters = None
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")
        
        # 优化器
        if self.adapters:
            self.optimizer = torch.optim.AdamW(
                self.adapters.parameters(),
                lr=1e-4,
                weight_decay=0.01,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-5,
                weight_decay=0.01,
            )
        
        logger.info(f"IncrementalTrainer initialized with strategy: {config.strategy}")
    
    def _compute_loss_with_regularization(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算带正则化的损失"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)
        
        # 标准损失
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        base_loss = outputs[0] if isinstance(outputs, tuple) else outputs["loss"]
        
        total_loss = base_loss
        losses = {"base_loss": base_loss.item()}
        
        # 添加EWC正则化
        if self.ewc and self.ewc.task_fisher:
            ewc_loss = self.ewc.compute_ewc_loss()
            total_loss = total_loss + ewc_loss
            losses["ewc_loss"] = ewc_loss.item()
        
        # 添加知识蒸馏损失
        if self.config.strategy in ["adapter", "elastic"]:
            with torch.no_grad():
                teacher_outputs = self.model.model(input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs["logits"] if isinstance(teacher_outputs, dict) else teacher_outputs[0]
            
            if self.adapters:
                student_outputs = self.adapters.base_model.model(input_ids, attention_mask=attention_mask)
                student_logits = student_outputs["logits"]
            else:
                student_logits = teacher_logits
            
            T = self.config.distill_temperature
            distill_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction='batchmean',
            ) * (T * T)
            
            total_loss = total_loss + self.config.distill_alpha * distill_loss
            losses["distill_loss"] = distill_loss.item()
        
        losses["total_loss"] = total_loss.item()
        
        return total_loss, losses
    
    def train_task(
        self,
        task_name: str,
        train_loader,
        num_epochs: int = 3,
        eval_loader = None,
    ):
        """训练新任务"""
        logger.info(f"Training task: {task_name}")
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch in train_loader:
                # 如果使用replay，混合新数据和旧数据
                if self.replay_buffer and len(self.replay_buffer.buffer) > 0:
                    replay_batch_size = int(batch["input_ids"].size(0) * self.config.replay_ratio)
                    
                    if replay_batch_size > 0:
                        replay_samples = self.replay_buffer.sample(replay_batch_size)
                        # 合并replay样本到batch...
                
                # 计算损失
                loss, losses = self._compute_loss_with_regularization(batch)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_losses.append(losses["total_loss"])
            
            avg_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
        
        # 训练后注册任务（如果是EWC策略）
        if self.ewc:
            self.ewc.register_task(task_name, train_loader, device=self.device)
        
        # 如果使用replay，保存样本
        if self.replay_buffer:
            for batch in train_loader:
                samples = [
                    {"input_ids": ids.tolist(), "labels": lbls.tolist()}
                    for ids, lbls in zip(batch["input_ids"], batch.get("labels", batch["input_ids"]))
                ]
                self.replay_buffer.add(samples, task_name)
        
        logger.info(f"Task {task_name} training completed")
    
    def evaluate_retention(
        self,
        old_task_loaders: Dict[str, DataLoader],
    ) -> Dict[str, float]:
        """评估对旧任务的保持能力"""
        self.model.eval()
        
        retention_scores = {}
        
        for task_name, loader in old_task_loaders.items():
            losses = []
            
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch.get("labels", input_ids).to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs[0] if isinstance(outputs, tuple) else outputs["loss"]
                    losses.append(loss.item())
            
            retention_scores[task_name] = np.mean(losses)
        
        self.model.train()
        
        return retention_scores


def incremental_finetuning_example():
    """增量微调示例"""
    print("=" * 60)
    print("Incremental Fine-tuning Example")
    print("=" * 60)
    
    print("""
    # 1. 使用Experience Replay
    from nexus_llm.incremental import (
        IncrementalTrainer, 
        IncrementalConfig, 
        ExperienceReplayBuffer
    )
    
    config = IncrementalConfig(
        strategy="replay",
        replay_buffer_size=10000,
        replay_ratio=0.3,
    )
    
    trainer = IncrementalTrainer(model, tokenizer, config)
    
    # 训练任务1
    trainer.train_task("task_1", task1_loader)
    
    # 训练任务2
    trainer.train_task("task_2", task2_loader)
    
    # 保存replay buffer
    trainer.replay_buffer.save("buffers/replay.pt")
    
    
    # 2. 使用EWC正则化
    config = IncrementalConfig(
        strategy="regularization",
        ewc_lambda=5000,
    )
    
    trainer = IncrementalTrainer(model, tokenizer, config)
    
    # 训练任务时会自动注册Fisher信息
    trainer.train_task("task_1", task1_loader)
    trainer.train_task("task_2", task2_loader)
    
    # 评估旧任务保持
    retention = trainer.evaluate_retention({
        "task_1": task1_eval_loader,
    })
    print(f"Task 1 retention loss: {retention['task_1']:.4f}")
    
    
    # 3. 使用Adapter
    config = IncrementalConfig(
        strategy="adapter",
        adapter_hidden_dim=64,
        adapter_alpha=0.5,
    )
    
    trainer = IncrementalTrainer(model, tokenizer, config)
    
    # 只训练Adapter参数
    trainer.train_task("task_1", task1_loader)
    
    # 保存Adapter
    trainer.adapters.save_adapters("adapters/task1.pt")
    """)
