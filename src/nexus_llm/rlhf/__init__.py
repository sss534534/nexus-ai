"""
RLHF (Reinforcement Learning from Human Feedback) Training Module
包含 Reward Model 和 PPO 训练实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from tqdm import tqdm
import numpy as np

from . import NexusConfig, NexusForCausalLM, NexusModel
from ..data import NexusTokenizer


logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """RLHF训练配置"""
    
    # Reward Model配置
    reward_model_name: str = "reward_model"
    reward_hidden_size: int = 4096
    reward_num_layers: int = 2
    
    # PPO配置
    ppo_epochs: int = 4
    ppo_clip_eps: float = 0.2  # PPO clip epsilon
    ppo_entropy_coef: float = 0.01  # 熵正则化系数
    ppo_value_coef: float = 0.5  # Value损失系数
    ppo_max_grad_norm: float = 1.0
    ppo_gamma: float = 1.0  # 折扣因子
    ppo_lam: float = 0.95  # GAE lambda
    
    # 学习率
    ppo_learning_rate: float = 1e-5
    ppo_init_kl_coef: float = 0.02  # KL散度系数
    ppo_target_kl: float = 0.02  # 目标KL散度
    
    # 生成配置
    gen_min_length: int = 4
    gen_max_length: int = 512
    gen_temperature: float = 1.0
    gen_top_p: float = 1.0
    gen_top_k: int = 0
    
    # 参考模型
    ref_model_match: bool = True
    
    # 数据配置
    preference_data_path: str = "data/preferences.jsonl"
    batch_size: int = 4
    gradient_accumulation_steps: int = 2


class RewardModel(nn.Module):
    """
    Reward Model - 预测人类偏好评分
    基于预训练语言模型，在最后一层添加回归头
    """
    
    def __init__(self, base_model: NexusModel, config: RLHFConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # 冻结基础模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Reward预测头
        self.reward_head = nn.Sequential(
            nn.Linear(config.reward_hidden_size, config.reward_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.reward_hidden_size // 2, 1),
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化reward head权重"""
        for module in self.reward_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播，返回奖励分数
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            
        Returns:
            reward_scores: 奖励分数，形状 (batch_size,)
        """
        # 获取基础模型输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # 获取最后一层hidden states
        if isinstance(outputs, dict):
            hidden_states = outputs["hidden_states"][-1]
        else:
            hidden_states = outputs[0]
        
        # 取序列最后一个非padding位置的表示
        # 假设EOS token之后是reward信号
        seq_len = input_ids.size(1)
        
        # 使用最后一个token的表示
        # 对于更好的reward估计，可以使用平均或特殊池化
        last_token_hidden = hidden_states[:, -1, :]
        
        # 预测reward
        reward = self.reward_head(last_token_hidden).squeeze(-1)
        
        return reward
    
    def get_reward(
        self,
        sequences: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取序列的reward和value估计
        
        Returns:
            rewards: 奖励分数
            values: 状态价值估计
        """
        rewards = self.forward(sequences, attention_mask)
        
        # 对于PPO，同时返回value估计
        # 这里简化处理，实际可以用单独的价值网络
        values = rewards.detach()
        
        return rewards, values


class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) 训练器
    实现基于人类反馈的强化学习训练
    """
    
    def __init__(
        self,
        policy_model: NexusForCausalLM,  # 当前策略模型
        ref_model: NexusForCausalLM,      # 参考模型（KL散度约束）
        reward_model: RewardModel,       # 奖励模型
        tokenizer: NexusTokenizer,
        config: RLHFConfig,
        device: str = "cuda",
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # 冻结奖励模型
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        # 策略模型需要梯度
        for param in self.policy_model.parameters():
            param.requires_grad = True
        
        # 优化器 - 只优化策略模型
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.ppo_learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        # KL散度系数（自适应调整）
        self.kl_coef = config.ppo_init_kl_coef
        
        logger.info("PPO Trainer initialized")
    
    def compute_rewards(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        responses_only: bool = True,
    ) -> torch.Tensor:
        """
        计算序列的reward
        使用reward model对整个序列打分，然后对response部分assign reward
        """
        with torch.no_grad():
            # 获取reward model的输出
            rewards = self.reward_model(sequences, attention_mask)
        
        if responses_only:
            # 找到response开始的位置（通常是 EOS 之后）
            # 这里简化为：找到最后一个 EOS token 之后的位置
            batch_size = sequences.size(0)
            
            # 初始化response reward为0
            response_rewards = torch.zeros(batch_size, device=self.device)
            
            # 找到EOS位置
            eos_token_id = self.tokenizer.eos_token_id
            for i in range(batch_size):
                # 找到序列中最后一个EOS的位置
                eos_positions = (sequences[i] == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    # 在EOS之后assign reward
                    pass  # 简化处理，直接使用最后的reward
            
            # 简化：直接返回整个序列的reward
            # 实际应用中需要更精细的处理
            return rewards
        else:
            return rewards
    
    def compute_kl_divergence(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算策略模型和参考模型之间的KL散度
        用于约束策略不要偏离参考模型太远
        """
        # 获取策略模型的logits
        policy_outputs = self.policy_model.model(sequences, attention_mask=attention_mask)
        policy_logits = policy_outputs["logits"] if isinstance(policy_outputs, dict) else policy_outputs[0]
        
        # 获取参考模型的logits
        with torch.no_grad():
            ref_outputs = self.ref_model.model(sequences, attention_mask=attention_mask)
            ref_logits = ref_outputs["logits"] if isinstance(ref_outputs, dict) else ref_outputs[0]
        
        # 计算logits的KL散度
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        # KL(p||q) = sum(p * (log(p) - log(q)))
        kl_div = F.kl_div(
            policy_log_probs,
            ref_log_probs,
            reduction='batchmean',
            log_target=True,
        )
        
        return kl_div
    
    def generate_responses(
        self,
        prompts: List[str],
        max_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        使用策略模型生成responses
        """
        # Tokenize prompts
        prompt_inputs = self.tokenizer.batch_encode(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length // 2,
        )
        
        input_ids = torch.tensor(prompt_inputs["input_ids"], device=self.device)
        attention_mask = torch.tensor(prompt_inputs["attention_mask"], device=self.device)
        
        # 生成responses
        self.policy_model.eval()
        with torch.no_grad():
            responses_ids = self.policy_model.generate(
                input_ids,
                max_new_tokens=max_length // 2,
                temperature=self.config.gen_temperature,
                top_p=self.config.gen_top_p,
                top_k=self.config.gen_top_k,
                do_sample=True,
            )
        
        # Decode responses
        responses = []
        for i in range(len(prompts)):
            response_ids = responses_ids[i, input_ids.size(1):].cpu().tolist()
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(response_text)
        
        # 合并prompt和response作为完整序列
        full_sequences = responses_ids
        
        return full_sequences, attention_mask, responses
    
    def ppo_step(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_rewards: torch.Tensor,
        responses_only: bool = True,
    ) -> Dict[str, float]:
        """
        执行一次PPO更新步骤
        """
        # 前向传播获取新的log_probs
        outputs = self.policy_model.model(sequences, attention_mask=attention_mask)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        
        # 计算log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 计算新序列的log prob
        # 简化：直接使用最后一个token的log prob
        new_log_probs = log_probs[:, :-1, :].contiguous()
        old_log_probs_clipped = old_log_probs[:, :-1].contiguous()
        
        # 计算ratio: exp(log_prob_new - log_prob_old)
        ratio = torch.exp(new_log_probs.sum(dim=-1) - old_log_probs_clipped.sum(dim=-1))
        
        # 计算reward（从reward model）
        with torch.no_grad():
            rewards = self.reward_model(sequences, attention_mask)
        
        # 计算KL散度
        kl_div = self.compute_kl_divergence(sequences, attention_mask)
        
        # PPO clip loss
        surr1 = ratio * rewards.unsqueeze(-1).sum(dim=1)  # 简化的reward
        surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_eps, 1 + self.config.ppo_clip_eps) * rewards.unsqueeze(-1).sum(dim=1)
        
        ppo_loss = -torch.min(surr1, surr2).mean()
        
        # KL penalty
        kl_loss = self.kl_coef * kl_div
        
        # 总损失
        total_loss = ppo_loss + kl_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.ppo_max_grad_norm,
        )
        
        self.optimizer.step()
        
        # 自适应调整KL系数
        current_kl = kl_div.item()
        if current_kl > self.config.ppo_target_kl * 1.5:
            self.kl_coef *= 1.5
        elif current_kl < self.config.ppo_target_kl * 0.5:
            self.kl_coef /= 1.5
        
        return {
            "ppo_loss": ppo_loss.item(),
            "kl_div": kl_div.item(),
            "kl_coef": self.kl_coef,
            "total_loss": total_loss.item(),
        }
    
    def train_step(
        self,
        prompts: List[str],
    ) -> Dict[str, float]:
        """
        执行一次完整的训练步骤：生成 -> 计算reward -> PPO更新
        """
        # Step 1: 生成responses
        sequences, attention_mask, responses = self.generate_responses(prompts)
        
        # Step 2: 计算rewards
        rewards = self.compute_rewards(sequences, attention_mask)
        
        # Step 3: 获取old log probs（这里简化处理，实际需要记录生成时的log probs）
        with torch.no_grad():
            outputs = self.policy_model.model(sequences, attention_mask=attention_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            log_probs = F.log_softmax(logits, dim=-1)
            old_log_probs = log_probs.sum(dim=-1)
        
        # Step 4: PPO更新
        metrics = self.ppo_step(
            sequences,
            attention_mask,
            old_log_probs,
            rewards,
        )
        
        metrics["mean_reward"] = rewards.mean().item()
        metrics["num_samples"] = len(prompts)
        
        return metrics
    
    def train(
        self,
        train_prompts: List[str],
        num_epochs: int = 3,
        batch_size: int = 4,
    ):
        """
        执行完整的RLHF训练
        """
        logger.info(f"Starting RLHF training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_metrics = []
            
            # 打乱数据
            indices = torch.randperm(len(train_prompts))
            
            for i in tqdm(range(0, len(train_prompts), batch_size), desc=f"Epoch {epoch+1}"):
                batch_prompts = [train_prompts[idx] for idx in indices[i:i+batch_size]]
                
                metrics = self.train_step(batch_prompts)
                epoch_metrics.append(metrics)
            
            # 汇总epoch统计
            avg_metrics = {
                key: np.mean([m[key] for m in epoch_metrics])
                for key in epoch_metrics[0].keys()
            }
            
            logger.info(
                f"Epoch {epoch+1}: "
                f"reward={avg_metrics['mean_reward']:.3f}, "
                f"kl_div={avg_metrics['kl_div']:.4f}, "
                f"loss={avg_metrics['total_loss']:.4f}"
            )
        
        logger.info("RLHF training completed")


class RewardModelTrainer:
    """
    Reward Model 训练器
    用于训练预测人类偏好的奖励模型
    """
    
    def __init__(
        self,
        base_model: NexusModel,
        tokenizer: NexusTokenizer,
        config: RLHFConfig,
        device: str = "cuda",
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # 创建reward model
        self.reward_model = RewardModel(base_model, config).to(device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.reward_model.parameters(),
            lr=1e-5,
            weight_decay=0.01,
        )
        
        logger.info("Reward Model Trainer initialized")
    
    def compute_reward_loss(
        self,
        chosen_sequences: torch.Tensor,
        rejected_sequences: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算reward model的对比损失
        chosen序列的reward应该高于rejected序列
        
        采用Bradley-Terry模型：
        P(preferred) = sigmoid(reward_chosen - reward_rejected)
        Loss = -log(P(preferred))
        """
        # 获取chosen和rejected的reward
        chosen_rewards = self.reward_model(chosen_sequences, chosen_mask)
        rejected_rewards = self.reward_model(rejected_sequences, rejected_mask)
        
        # 计算reward差异
        reward_diff = chosen_rewards - rejected_rewards
        
        # 对比损失 (pairwise ranking loss)
        loss = -F.logsigmoid(reward_diff).mean()
        
        # 准确率：在多少比例下chosen的reward高于rejected
        accuracy = (reward_diff > 0).float().mean()
        
        return loss, accuracy
    
    def train_step(
        self,
        chosen_texts: List[str],
        rejected_texts: List[str],
    ) -> Dict[str, float]:
        """执行一次训练步骤"""
        # Tokenize
        chosen_inputs = self.tokenizer.batch_encode(chosen_texts)
        rejected_inputs = self.tokenizer.batch_encode(rejected_texts)
        
        chosen_ids = torch.tensor(chosen_inputs["input_ids"], device=self.device)
        chosen_mask = torch.tensor(chosen_inputs["attention_mask"], device=self.device)
        rejected_ids = torch.tensor(rejected_inputs["input_ids"], device=self.device)
        rejected_mask = torch.tensor(rejected_inputs["attention_mask"], device=self.device)
        
        # 前向传播
        loss, accuracy = self.compute_reward_loss(
            chosen_ids, rejected_ids, chosen_mask, rejected_mask
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
        }
    
    def train(
        self,
        preference_data: List[Dict[str, Any]],
        num_epochs: int = 3,
        batch_size: int = 4,
        save_path: Optional[str] = None,
    ):
        """
        训练Reward Model
        
        Args:
            preference_data: 人类偏好数据，格式为 [{"chosen": "text1", "rejected": "text2"}, ...]
        """
        logger.info(f"Training Reward Model on {len(preference_data)} samples")
        
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_accs = []
            
            for i in tqdm(range(0, len(preference_data), batch_size), desc=f"Epoch {epoch+1}"):
                batch = preference_data[i:i+batch_size]
                chosen_texts = [item["chosen"] for item in batch]
                rejected_texts = [item["rejected"] for item in batch]
                
                metrics = self.train_step(chosen_texts, rejected_texts)
                epoch_losses.append(metrics["loss"])
                epoch_accs.append(metrics["accuracy"])
            
            avg_loss = np.mean(epoch_losses)
            avg_acc = np.mean(epoch_accs)
            
            logger.info(
                f"Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={avg_acc:.3f}"
            )
            
            # 保存checkpoint
            if save_path:
                self.save(save_path)
        
        logger.info("Reward Model training completed")
    
    def save(self, save_path: str):
        """保存reward model"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(
            self.reward_model.state_dict(),
            save_path / "reward_model.pt"
        )
        
        logger.info(f"Reward model saved to {save_path}")
    
    def load(self, load_path: str):
        """加载reward model"""
        load_path = Path(load_path)
        
        self.reward_model.load_state_dict(
            torch.load(load_path / "reward_model.pt")
        )
        
        logger.info(f"Reward model loaded from {load_path}")


class PreferenceDataset(Dataset):
    """
    人类偏好数据集
    存储 (prompt, chosen_response, rejected_response) 三元组
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: NexusTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.data)} preference samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        return {
            "prompt": item.get("prompt", ""),
            "chosen": item.get("chosen", item.get("preferred", "")),
            "rejected": item.get("rejected", item.get("rejected", "")),
            "score_diff": item.get("score_diff", 1.0),
        }
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate函数"""
        prompts = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        rejected = [item["rejected"] for item in batch]
        
        # Tokenize
        chosen_enc = self.tokenizer.batch_encode(chosen)
        rejected_enc = self.tokenizer.batch_encode(rejected)
        
        return {
            "prompt": prompts,
            "chosen_ids": torch.tensor(chosen_enc["input_ids"]),
            "chosen_mask": torch.tensor(chosen_enc["attention_mask"]),
            "rejected_ids": torch.tensor(rejected_enc["input_ids"]),
            "rejected_mask": torch.tensor(rejected_enc["attention_mask"]),
            "score_diff": torch.tensor([item["score_diff"] for item in batch]),
        }


def create_rlhf_pipeline(
    policy_model: NexusForCausalLM,
    tokenizer: NexusTokenizer,
    rlhf_config: RLHFConfig,
    device: str = "cuda",
) -> Tuple[PPOTrainer, RewardModelTrainer]:
    """
    创建完整的RLHF训练管道
    
    Returns:
        ppo_trainer: PPO训练器
        reward_trainer: Reward Model训练器
    """
    # 创建参考模型（Policy model的克隆）
    ref_model = NexusForCausalLM(policy_model.config)
    ref_model.load_state_dict(policy_model.state_dict())
    ref_model = ref_model.to(device)
    
    # 创建Reward Model
    reward_trainer = RewardModelTrainer(
        base_model=policy_model.model,
        tokenizer=tokenizer,
        config=rlhf_config,
        device=device,
    )
    
    # 创建PPO Trainer
    ppo_trainer = PPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_model=reward_trainer.reward_model,
        tokenizer=tokenizer,
        config=rlhf_config,
        device=device,
    )
    
    return ppo_trainer, reward_trainer


def train_reward_model_example():
    """Reward Model训练示例"""
    print("=" * 60)
    print("Reward Model Training Example")
    print("=" * 60)
    
    print("""
    # 1. 准备偏好数据
    # 格式: [{"chosen": "好的回答", "rejected": "差的回答"}, ...]
    
    # 2. 初始化训练器
    from nexus_llm.rlhf import RewardModelTrainer, RLHFConfig
    
    config = RLHFConfig(
        preference_data_path="data/preferences.jsonl",
        batch_size=8,
    )
    
    trainer = RewardModelTrainer(
        base_model=policy_model.model,
        tokenizer=tokenizer,
        config=config,
    )
    
    # 3. 训练
    trainer.train(
        preference_data=preference_data,
        num_epochs=3,
        save_path="checkpoints/reward_model",
    )
    """)


def rlhf_training_example():
    """RLHF训练示例"""
    print("\n" + "=" * 60)
    print("RLHF Training Example")
    print("=" * 60)
    
    print("""
    # 1. 初始化RLHF管道
    from nexus_llm.rlhf import create_rlhf_pipeline
    
    ppo_trainer, reward_trainer = create_rlhf_pipeline(
        policy_model=policy_model,
        tokenizer=tokenizer,
        rlhf_config=config,
    )
    
    # 2. 先训练Reward Model（如果还没有）
    reward_trainer.train(preference_data)
    
    # 3. PPO训练
    ppo_trainer.train(
        train_prompts=prompts,
        num_epochs=3,
        batch_size=4,
    )
    
    # 4. 保存微调后的模型
    policy_model.save_pretrained("checkpoints/rlhf_model")
    """)
