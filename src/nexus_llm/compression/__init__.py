"""
Model Compression Module - 模型压缩优化
包含知识蒸馏、剪枝、量化等压缩技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import numpy as np
from tqdm import tqdm

from . import NexusConfig, NexusForCausalLM, NexusModel


logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """知识蒸馏配置"""
    temperature: float = 2.0  # 温度参数
    alpha: float = 0.5  # 蒸馏损失权重
    beta: float = 0.5  # 隐藏层损失权重
    
    # 对齐策略
    hidden_match: bool = True
    attention_match: bool = True
    
    # 层映射
    layer_mapping: str = "last_k"  # last_k, mid, all
    num_layers_to_match: int = 4


@dataclass
class PruningConfig:
    """剪枝配置"""
    sparsity: float = 0.5  # 稀疏度
    pruning_type: str = "magnitude"  # magnitude, random, gradient
    
    # 结构化剪枝
    structured: bool = True
    prune_heads: bool = False  # 剪枝注意力头
    
    # 迭代剪枝
    iterative: bool = True
    iterations: int = 3
    sparsity_per_iteration: float = 0.2


@dataclass
class QuantizationConfig:
    """量化配置"""
    quantization_type: str = "dynamic"  # dynamic, static, qat
    
    # 量化精度
    bits: int = 8  # 4, 8, 16
    
    # 量化策略
    per_channel: bool = False
    per_token: bool = False
    
    # 校准
    calibration_data: Optional[str] = None
    calibration_samples: int = 512


class KnowledgeDistiller:
    """
    知识蒸馏器
    将大模型（Teacher）的知识迁移到小模型（Student）
    """
    
    def __init__(
        self,
        teacher_model: NexusForCausalLM,
        student_model: NexusForCausalLM,
        tokenizer: Any,
        config: DistillationConfig,
        device: str = "cuda",
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # 冻结教师模型
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # 学生模型优化器
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=1e-4,
            weight_decay=0.01,
        )
        
        logger.info("Knowledge Distiller initialized")
    
    def compute_student_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """计算学生模型的标准交叉熵损失"""
        # Shift for causal LM
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """计算蒸馏损失（KL散度）"""
        T = self.config.temperature
        
        # 应用温度缩放
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        
        # KL散度损失
        distill_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
        ) * (T * T)  # 补偿温度因子的缩放
        
        return distill_loss
    
    def hidden_states_matching(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """匹配隐藏层表示"""
        # MSE损失对齐隐藏状态
        loss = F.mse_loss(student_hidden, teacher_hidden)
        return loss
    
    def forward_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """前向传播计算损失"""
        batch_size = input_ids.size(0)
        
        # 教师模型前向传播
        with torch.no_grad():
            teacher_outputs = self.teacher.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            teacher_logits = teacher_outputs["logits"]
            teacher_hidden = teacher_outputs["hidden_states"][-1]
        
        # 学生模型前向传播
        student_outputs = self.student.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        student_logits = student_outputs["logits"]
        student_hidden = student_outputs["hidden_states"][-1]
        
        # 计算各项损失
        student_loss = self.compute_student_loss(student_logits, labels)
        distill_loss = self.compute_distillation_loss(student_logits, teacher_logits)
        
        # 总损失
        total_loss = (
            self.config.alpha * student_loss +
            (1 - self.config.alpha) * distill_loss
        )
        
        return {
            "total_loss": total_loss,
            "student_loss": student_loss,
            "distill_loss": distill_loss,
        }
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """单步训练"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        self.optimizer.zero_grad()
        
        losses = self.forward_pass(input_ids, attention_mask, labels)
        losses["total_loss"].backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.student.parameters(),
            max_norm=1.0,
        )
        
        self.optimizer.step()
        
        return {
            key: value.item()
            for key, value in losses.items()
        }
    
    def distill(
        self,
        train_loader: DataLoader,
        num_epochs: int = 3,
        save_path: Optional[str] = None,
    ):
        """执行蒸馏训练"""
        logger.info(f"Starting distillation for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                metrics = self.train_step(batch)
                epoch_losses.append(metrics["total_loss"])
            
            avg_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
            
            if save_path:
                self.student.save_pretrained(save_path)
        
        logger.info("Distillation completed")
        
        return self.student


class StructuredPruner:
    """
    结构化剪枝器
    支持注意力头剪枝和FFN层剪枝
    """
    
    def __init__(
        self,
        model: NexusForCausalLM,
        config: PruningConfig,
    ):
        self.model = model
        self.config = config
        self.mask_cache: Dict[str, torch.Tensor] = {}
    
    def compute_importance_scores(self) -> Dict[str, torch.Tensor]:
        """
        计算参数重要性分数
        使用权重幅值作为重要性指标
        """
        importance = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # 使用权重幅值作为重要性分数
                importance[name] = param.data.abs().mean()
        
        return importance
    
    def compute_gradient_importance(self) -> Dict[str, torch.Tensor]:
        """
        使用梯度计算重要性分数
        """
        importance = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None and 'weight' in name:
                # 梯度与权重的乘积作为重要性
                importance[name] = (param.data.abs() * param.grad.abs()).mean()
        
        return importance
    
    def prune_attention_heads(
        self,
        importance_threshold: float,
    ) -> torch.Tensor:
        """
        剪枝不重要的注意力头
        返回每个头是否被保留的掩码
        """
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        
        head_importance = torch.zeros(num_layers, num_heads)
        
        # 计算每个头的重要性
        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]
            
            # 使用注意力权重的方差作为重要性
            # 这里简化处理，实际需要更复杂的计算
            head_importance[layer_idx] = torch.rand(num_heads)  # 模拟
        
        # 根据阈值决定保留哪些头
        head_mask = head_importance > importance_threshold
        
        return head_mask
    
    def apply_structured_pruning(
        self,
        sparsity: float,
    ) -> NexusForCausalLM:
        """
        应用结构化剪枝
        """
        logger.info(f"Applying structured pruning with sparsity={sparsity}")
        
        # 计算重要性分数
        importance = self.compute_importance_scores()
        
        # 找出要剪枝的参数
        total_params = 0
        pruned_params = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' not in name:
                continue
            
            total_params += param.numel()
            
            # 计算该层的稀疏度目标
            param_importance = importance.get(name, torch.ones_like(param))
            
            # 确定阈值
            threshold = torch.quantile(
                param_importance.flatten(),
                sparsity,
            )
            
            # 创建掩码
            mask = (param_importance > threshold).float()
            
            # 应用掩码
            param.data = param.data * mask
            
            pruned_params += (mask == 0).sum().item()
            
            self.mask_cache[name] = mask
        
        sparsity_achieved = pruned_params / total_params
        logger.info(f"Pruning complete: {sparsity_achieved:.2%} sparsity achieved")
        
        return self.model
    
    def iterative_prune(
        self,
        target_sparsity: float,
    ) -> NexusForCausalLM:
        """
        迭代剪枝
        """
        if not self.config.iterative:
            return self.apply_structured_pruning(target_sparsity)
        
        current_sparsity = 0
        sparsity_per_iter = target_sparsity / self.config.iterations
        
        for i in range(self.config.iterations):
            current_sparsity += sparsity_per_iter
            logger.info(f"Iteration {i+1}: target sparsity = {current_sparsity:.2%}")
            
            self.model = self.apply_structured_pruning(current_sparsity)
        
        return self.model
    
    def get_pruning_mask(self, name: str) -> Optional[torch.Tensor]:
        """获取指定参数的剪枝掩码"""
        return self.mask_cache.get(name)
    
    def restore_weights(self):
        """恢复被剪枝的权重（需要保存原始权重）"""
        logger.info("Restoring pruned weights not implemented")


class ModelCompressor:
    """
    模型压缩管理器
    整合蒸馏、剪枝、量化
    """
    
    def __init__(
        self,
        model: NexusForCausalLM,
        tokenizer: Any,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def compress(
        self,
        method: str = "quantization",
        **kwargs,
    ) -> NexusForCausalLM:
        """
        执行模型压缩
        
        Args:
            method: 压缩方法 ("distillation", "pruning", "quantization")
        """
        if method == "distillation":
            return self._distill(**kwargs)
        elif method == "pruning":
            return self._prune(**kwargs)
        elif method == "quantization":
            return self._quantize(**kwargs)
        elif method == "all":
            return self._compress_all(**kwargs)
        else:
            raise ValueError(f"Unknown compression method: {method}")
    
    def _distill(
        self,
        teacher_model: NexusForCausalLM,
        train_loader: DataLoader,
        config: DistillationConfig,
        num_epochs: int = 3,
    ) -> NexusForCausalLM:
        """知识蒸馏"""
        distiller = KnowledgeDistiller(
            teacher_model=teacher_model,
            student_model=self.model,
            tokenizer=self.tokenizer,
            config=config,
            device=self.device,
        )
        
        return distiller.distill(train_loader, num_epochs)
    
    def _prune(
        self,
        config: PruningConfig,
        target_sparsity: float = 0.5,
    ) -> NexusForCausalLM:
        """模型剪枝"""
        pruner = StructuredPruner(self.model, config)
        
        if config.iterative:
            return pruner.iterative_prune(target_sparsity)
        else:
            return pruner.apply_structured_pruning(target_sparsity)
    
    def _quantize(
        self,
        config: QuantizationConfig,
    ) -> NexusForCausalLM:
        """模型量化"""
        logger.info(f"Quantizing model to {config.bits} bits")
        
        if config.quantization_type == "dynamic":
            return self._dynamic_quantization()
        elif config.quantization_type == "static":
            return self._static_quantization(config)
        elif config.quantization_type == "qat":
            return self._quantization_aware_training(config)
        else:
            raise ValueError(f"Unknown quantization type: {config.quantization_type}")
    
    def _dynamic_quantization(self) -> NexusForCausalLM:
        """动态量化"""
        from torch.quantization import quantize_dynamic
        
        # 动态量化Linear层
        self.model = quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        
        logger.info("Dynamic quantization applied")
        return self.model
    
    def _static_quantization(self, config: QuantizationConfig):
        """静态量化"""
        from torch.quantization import quantize_static
        
        # 设置量化配置
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 准备量化
        torch.quantization.prepare(self.model, inplace=True)
        
        # 校准（需要运行一些样本）
        # calibrate(self.model, calibration_loader)
        
        # 转换
        self.model = torch.quantization.convert(self.model, inplace=True)
        
        logger.info("Static quantization applied")
        return self.model
    
    def _quantization_aware_training(self, config: QuantizationConfig):
        """量化感知训练"""
        from torch.quantization import prepare_qat, convert
        
        self.model.train()
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        prepare_qat(self.model, inplace=True)
        
        # QAT训练步骤...
        
        self.model.eval()
        convert(self.model, inplace=True)
        
        logger.info("QAT quantization applied")
        return self.model
    
    def _compress_all(
        self,
        config: Dict[str, Any],
    ) -> NexusForCausalLM:
        """完整压缩流程"""
        # 1. 先蒸馏
        if "distillation" in config:
            self.model = self.compress("distillation", **config["distillation"])
        
        # 2. 再剪枝
        if "pruning" in config:
            self.model = self.compress("pruning", **config["pruning"])
        
        # 3. 最后量化
        if "quantization" in config:
            self.model = self.compress("quantization", **config["quantization"])
        
        return self.model
    
    def estimate_compression_ratio(self) -> Dict[str, float]:
        """估算压缩效果"""
        # 原始模型大小
        original_size = sum(
            p.numel() * p.element_size()
            for p in self.model.parameters()
        )
        
        # 估算压缩后大小（简化估算）
        # 实际压缩比取决于具体的压缩方法
        
        return {
            "original_size_mb": original_size / (1024 ** 2),
            "estimated_quantized_8bit_mb": original_size * 0.25 / (1024 ** 2),
            "estimated_quantized_4bit_mb": original_size * 0.125 / (1024 ** 2),
        }


def prune_and_quantize_example():
    """剪枝和量化示例"""
    print("=" * 60)
    print("Model Compression Example")
    print("=" * 60)
    
    print("""
    # 1. 知识蒸馏
    from nexus_llm.compression import KnowledgeDistiller, DistillationConfig
    
    teacher = load_model("teacher_model")
    student = create_student_model()
    
    config = DistillationConfig(
        temperature=2.0,
        alpha=0.5,
    )
    
    distiller = KnowledgeDistiller(teacher, student, tokenizer, config)
    student = distiller.distill(train_loader, num_epochs=3)
    
    # 2. 结构化剪枝
    from nexus_llm.compression import StructuredPruner, PruningConfig
    
    pruning_config = PruningConfig(
        sparsity=0.5,
        structured=True,
        iterative=True,
        iterations=3,
    )
    
    pruner = StructuredPruner(model, pruning_config)
    model = pruner.iterative_prune(target_sparsity=0.5)
    
    # 3. 量化
    from nexus_llm.compression import ModelCompressor, QuantizationConfig
    
    quant_config = QuantizationConfig(
        quantization_type="dynamic",
        bits=8,
    )
    
    compressor = ModelCompressor(model, tokenizer)
    quantized_model = compressor.compress("quantization", config=quant_config)
    
    # 4. 完整压缩流程
    compression_config = {
        "quantization": {"quantization_type": "dynamic", "bits": 8},
        "pruning": {"sparsity": 0.3, "iterative": True},
    }
    
    compressed_model = compressor.compress("all", config=compression_config)
    
    # 5. 查看压缩效果
    stats = compressor.estimate_compression_ratio()
    print(f"Original: {stats['original_size_mb']:.1f} MB")
    print(f"8-bit Quantized: {stats['estimated_quantized_8bit_mb']:.1f} MB")
    """)
