"""
集成测试 - 端到端训练→推理流程验证
测试整个框架的核心功能链路
"""

import os
import sys
import pytest
import tempfile
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nexus_llm import (
    NexusConfig,
    NexusForCausalLM,
    NexusTokenizer,
    DataConfig,
    TrainingConfig,
    Trainer,
    InferenceEngine,
    InferenceConfig,
    create_rlhf_pipeline,
    RLHFConfig,
)


# ============================================================================
# 测试配置
# ============================================================================
class TestConfig:
    """小规模测试配置"""
    vocab_size = 1000
    hidden_size = 128
    intermediate_size = 256
    num_hidden_layers = 2
    num_attention_heads = 4
    num_key_value_heads = 2
    max_position_embeddings = 128
    
    @classmethod
    def get_model_config(cls):
        return NexusConfig(
            vocab_size=cls.vocab_size,
            hidden_size=cls.hidden_size,
            intermediate_size=cls.intermediate_size,
            num_hidden_layers=cls.num_hidden_layers,
            num_attention_heads=cls.num_attention_heads,
            num_key_value_heads=cls.num_key_value_heads,
            max_position_embeddings=cls.max_position_embeddings,
        )


# ============================================================================
# 模型测试
# ============================================================================
class TestModelCreation:
    """测试模型创建和基本功能"""
    
    def test_model_creation(self):
        """测试模型可以正常创建"""
        config = TestConfig.get_model_config()
        model = NexusForCausalLM(config)
        
        assert model is not None
        assert model.config.vocab_size == TestConfig.vocab_size
    
    def test_model_forward(self):
        """测试模型前向传播"""
        config = TestConfig.get_model_config()
        model = NexusForCausalLM(config)
        
        # 创建输入
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        
        # 前向传播
        outputs = model(input_ids)
        
        assert outputs is not None
        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 16, config.vocab_size)
    
    def test_model_generate(self):
        """测试模型生成功能"""
        config = TestConfig.get_model_config()
        model = NexusForCausalLM(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=1.0,
                top_k=50,
            )
        
        assert generated is not None
        assert generated.shape[1] >= input_ids.shape[1]
    
    def test_model_save_load(self):
        """测试模型保存和加载"""
        config = TestConfig.get_model_config()
        model = NexusForCausalLM(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存
            save_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), save_path)
            
            assert save_path.exists()
            
            # 加载
            new_model = NexusForCausalLM(config)
            new_model.load_state_dict(torch.load(save_path))
            
            # 验证参数一致
            for (name1, param1), (name2, param2) in zip(
                model.named_parameters(), new_model.named_parameters()
            ):
                assert torch.allclose(param1, param2)


# ============================================================================
# 数据处理测试
# ============================================================================
class TestDataProcessing:
    """测试数据处理流程"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        return [
            {"text": "这是一个测试句子。", "source": "test"},
            {"text": "Another test sentence.", "source": "test"},
            {"text": "人工智能是计算机科学的一个分支。", "source": "test"},
        ]
    
    def test_data_config(self):
        """测试数据配置"""
        config = DataConfig(
            tokenizer_path="dummy",
            max_seq_length=128,
        )
        
        assert config.max_seq_length == 128
    
    def test_data_format(self, sample_data):
        """测试数据格式"""
        for item in sample_data:
            assert "text" in item
            assert len(item["text"]) > 0


# ============================================================================
# 训练测试
# ============================================================================
class TestTraining:
    """测试训练流程"""
    
    @pytest.fixture
    def training_setup(self):
        """创建训练设置"""
        config = TestConfig.get_model_config()
        model = NexusForCausalLM(config)
        
        # 创建简单数据
        train_data = [
            {"input_ids": torch.randint(0, config.vocab_size, (16,))}
            for _ in range(10)
        ]
        
        return model, train_data
    
    def test_training_config(self):
        """测试训练配置"""
        config = TrainingConfig(
            learning_rate=1e-4,
            num_epochs=1,
            batch_size=2,
        )
        
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 1
    
    def test_training_step(self, training_setup):
        """测试单步训练"""
        model, train_data = training_setup
        
        # 设置为训练模式
        model.train()
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 单步训练
        batch = train_data[0]
        input_ids = batch["input_ids"].unsqueeze(0)
        
        outputs = model(input_ids)
        loss = outputs["logits"].mean()  # 简单损失
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        assert loss.item() is not None
    
    def test_gradient_checkpointing(self):
        """测试梯度检查点"""
        config = TestConfig.get_model_config()
        config.gradient_checkpointing = True
        
        model = NexusForCausalLM(config)
        
        # 验证梯度检查点已启用
        assert model.config.gradient_checkpointing


# ============================================================================
# 推理测试
# ============================================================================
class TestInference:
    """测试推理流程"""
    
    @pytest.fixture
    def inference_setup(self):
        """创建推理设置"""
        config = TestConfig.get_model_config()
        model = NexusForCausalLM(config)
        model.eval()
        
        return model
    
    def test_inference_config(self):
        """测试推理配置"""
        config = InferenceConfig(
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.95,
        )
        
        assert config.max_new_tokens == 50
        assert config.temperature == 0.7
    
    def test_batch_inference(self, inference_setup):
        """测试批量推理"""
        model = inference_setup
        
        # 批量输入
        batch_size = 4
        input_ids = torch.randint(0, TestConfig.vocab_size, (batch_size, 8))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs["logits"].shape[0] == batch_size
    
    def test_streaming_inference(self, inference_setup):
        """测试流式推理"""
        model = inference_setup
        
        input_ids = torch.randint(0, TestConfig.vocab_size, (1, 8))
        
        generated_tokens = []
        with torch.no_grad():
            for _ in range(5):
                outputs = model(input_ids)
                next_token = outputs["logits"][:, -1, :].argmax(dim=-1)
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        assert len(generated_tokens) == 5


# ============================================================================
# RLHF 测试
# ============================================================================
class TestRLHF:
    """测试 RLHF 流程"""
    
    def test_rlhf_config(self):
        """测试 RLHF 配置"""
        config = RLHFConfig(
            ppo_clip_eps=0.2,
            ppo_learning_rate=1e-5,
        )
        
        assert config.ppo_clip_eps == 0.2
    
    def test_rlhf_pipeline_creation(self):
        """测试 RLHF 管道创建"""
        config = TestConfig.get_model_config()
        
        rlhf_config = RLHFConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
        )
        
        pipeline = create_rlhf_pipeline(rlhf_config)
        
        assert pipeline is not None
        assert pipeline.reward_model is not None
        assert pipeline.ppo_trainer is not None


# ============================================================================
# 端到端测试
# ============================================================================
class TestEndToEnd:
    """端到端流程测试"""
    
    def test_full_pipeline(self):
        """测试完整流程：创建→训练→推理"""
        # 1. 创建模型
        config = TestConfig.get_model_config()
        model = NexusForCausalLM(config)
        
        # 2. 训练几步
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        for _ in range(3):
            input_ids = torch.randint(0, config.vocab_size, (2, 16))
            outputs = model(input_ids)
            loss = outputs["logits"].mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # 3. 推理
        model.eval()
        with torch.no_grad():
            input_ids = torch.randint(0, config.vocab_size, (1, 8))
            outputs = model(input_ids)
            generated = model.generate(input_ids, max_new_tokens=5)
        
        assert generated is not None
        assert generated.shape[1] > input_ids.shape[1]
    
    def test_save_load_pipeline(self):
        """测试保存加载流程"""
        config = TestConfig.get_model_config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建并保存
            model1 = NexusForCausalLM(config)
            model1.train()
            
            # 训练一步
            optimizer = torch.optim.Adam(model1.parameters(), lr=1e-4)
            input_ids = torch.randint(0, config.vocab_size, (2, 16))
            outputs = model1(input_ids)
            loss = outputs["logits"].mean()
            loss.backward()
            optimizer.step()
            
            # 保存
            save_path = Path(tmpdir) / "checkpoint.pt"
            torch.save({
                "model_state_dict": model1.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_path)
            
            # 加载到新模型
            model2 = NexusForCausalLM(config)
            checkpoint = torch.load(save_path)
            model2.load_state_dict(checkpoint["model_state_dict"])
            
            # 验证输出一致
            model1.eval()
            model2.eval()
            
            test_input = torch.randint(0, config.vocab_size, (1, 8))
            
            with torch.no_grad():
                out1 = model1(test_input)
                out2 = model2(test_input)
            
            assert torch.allclose(out1["logits"], out2["logits"])


# ============================================================================
# 性能基准测试
# ============================================================================
class TestPerformance:
    """性能基准测试"""
    
    def test_inference_latency(self):
        """测试推理延迟"""
        config = TestConfig.get_model_config()
        model = NexusForCausalLM(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                model(input_ids)
        
        # 测量
        import time
        start = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                model(input_ids)
        
        elapsed = time.time() - start
        latency_ms = elapsed / 10 * 1000
        
        print(f"\nInference latency: {latency_ms:.2f} ms")
        
        # 基本性能要求
        assert latency_ms < 100  # 小模型应该很快
    
    def test_memory_usage(self):
        """测试内存使用"""
        config = TestConfig.get_model_config()
        model = NexusForCausalLM(config)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"\nTotal parameters: {total_params:,}")
        
        # 小模型应该参数较少
        assert total_params < 1e6  # 测试配置应该小于1M参数
    
    def test_generation_speed(self):
        """测试生成速度"""
        config = TestConfig.get_model_config()
        model = NexusForCausalLM(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        
        import time
        start = time.time()
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=20,
            )
        
        elapsed = time.time() - start
        tokens_per_sec = 20 / elapsed
        
        print(f"\nGeneration speed: {tokens_per_sec:.2f} tokens/sec")


# ============================================================================
# 运行测试
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])