"""
Example: Vertical Domain Training and Deployment
"""

import json
from pathlib import Path

from nexus_llm import NexusConfig, NexusForCausalLM
from nexus_llm.data import NexusTokenizer, DataConfig, VerticalDomainProcessor
from nexus_llm.training import Trainer, TrainingConfig
from nexus_llm.serving import InferenceEngine, InferenceConfig


def create_finance_data():
    """Create sample finance domain training data."""
    
    finance_data = [
        {
            "context": "某公司股票代码为600519，主营业务为白酒生产和销售。",
            "question": "这家公司的主营业务是什么？",
            "answer": "这家公司的主营业务是白酒生产和销售。"
        },
        {
            "context": "2024年第一季度，GDP同比增长5.3%，CPI同比上涨0.1%。",
            "question": "2024年Q1 GDP增速是多少？",
            "answer": "2024年第一季度GDP同比增长5.3%。"
        },
        {
            "context": "央行宣布下调存款准备金率0.5个百分点，释放长期资金约1万亿元。",
            "question": "降准释放了多少资金？",
            "answer": "此次降准释放长期资金约1万亿元。"
        },
    ]
    
    return finance_data


def create_medical_data():
    """Create sample medical domain training data."""
    
    medical_data = [
        {
            "symptoms": "患者发热38.5度，咳嗽，乏力，已持续3天。",
            "diagnosis": "上呼吸道感染",
            "treatment": "建议休息，多饮水，必要时服用退烧药。"
        },
        {
            "symptoms": "头痛，恶心，视力模糊，血压160/100mmHg。",
            "diagnosis": "高血压危象",
            "treatment": "立即降压治疗，监测生命体征，完善检查。"
        },
    ]
    
    return medical_data


def create_legal_data():
    """Create sample legal domain training data."""
    
    legal_data = [
        {
            "case": "甲公司与乙公司签订买卖合同，约定交货日期为2024年3月1日。乙公司逾期未交货。",
            "question": "甲公司可以主张什么权利？",
            "analysis": "根据《民法典》第577条，乙公司构成违约，甲公司可以要求继续履行、采取补救措施或者赔偿损失。"
        },
        {
            "case": "员工张某在试用期被公司以不符合录用条件为由解除劳动合同。",
            "question": "公司解除劳动合同是否合法？",
            "analysis": "根据《劳动合同法》第39条，用人单位在试用期解除劳动合同需证明劳动者不符合录用条件，否则构成违法解除。"
        },
    ]
    
    return legal_data


def finance_domain_example():
    """Finance domain training example."""
    
    print("=" * 60)
    print("Finance Domain Training Example")
    print("=" * 60)
    
    # Create data
    print("\n1. Creating finance domain data...")
    finance_data = create_finance_data()
    
    # Save to file
    data_dir = Path("data/finance")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    with open(data_dir / "train.jsonl", "w") as f:
        for item in finance_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"   Created {len(finance_data)} samples")
    
    # Setup processor
    print("\n2. Setting up domain processor...")
    data_config = DataConfig(
        train_file=str(data_dir / "train.jsonl"),
        max_seq_length=2048,
    )
    
    tokenizer = NexusTokenizer(data_config)
    processor = VerticalDomainProcessor(
        config=data_config,
        tokenizer=tokenizer,
        domain="finance",
    )
    
    # Format data
    print("\n3. Formatting data...")
    formatted_data = []
    for item in finance_data:
        formatted = processor._format_finance(item)
        formatted_data.append(formatted)
        print(f"   Example: {formatted[:100]}...")
    
    print("\n4. Training configuration...")
    model_config = NexusConfig(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=16,
        vocab_size=100000,
    )
    
    training_config = TrainingConfig(
        learning_rate=5e-5,
        batch_size=2,
        max_steps=1000,
        output_dir="outputs/finance-model",
    )
    
    print(f"   Model size: {model_config.hidden_size} hidden dim")
    print(f"   Learning rate: {training_config.learning_rate}")
    print(f"   Output: {training_config.output_dir}")
    
    print("\n5. To start training, run:")
    print("   nexus-train --model-config configs/model_config.yaml \\")
    print("       --training-config configs/training_config.yaml \\")
    print("       --data-config configs/data_config.yaml")


def medical_domain_example():
    """Medical domain training example."""
    
    print("\n" + "=" * 60)
    print("Medical Domain Training Example")
    print("=" * 60)
    
    # Create data
    print("\n1. Creating medical domain data...")
    medical_data = create_medical_data()
    
    data_dir = Path("data/medical")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    with open(data_dir / "train.jsonl", "w") as f:
        for item in medical_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"   Created {len(medical_data)} samples")
    
    # Show formatting
    print("\n2. Data formatting example:")
    data_config = DataConfig()
    tokenizer = NexusTokenizer(data_config)
    processor = VerticalDomainProcessor(
        config=data_config,
        tokenizer=tokenizer,
        domain="medical",
    )
    
    for item in medical_data[:1]:
        formatted = processor._format_medical(item)
        print(f"   {formatted}")
    
    print("\n3. Note: Medical domain requires:")
    print("   - HIPAA compliance for US data")
    print("   - Expert review of training data")
    print("   - Clear disclaimer for AI-generated advice")


def legal_domain_example():
    """Legal domain training example."""
    
    print("\n" + "=" * 60)
    print("Legal Domain Training Example")
    print("=" * 60)
    
    # Create data
    print("\n1. Creating legal domain data...")
    legal_data = create_legal_data()
    
    data_dir = Path("data/legal")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    with open(data_dir / "train.jsonl", "w") as f:
        for item in legal_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"   Created {len(legal_data)} samples")
    
    # Show formatting
    print("\n2. Data formatting example:")
    data_config = DataConfig()
    tokenizer = NexusTokenizer(data_config)
    processor = VerticalDomainProcessor(
        config=data_config,
        tokenizer=tokenizer,
        domain="legal",
    )
    
    for item in legal_data[:1]:
        formatted = processor._format_legal(item)
        print(f"   {formatted}")
    
    print("\n3. Note: Legal domain requires:")
    print("   - Jurisdiction-specific training data")
    print("   - Regular updates for law changes")
    print("   - Clear disclaimer: not legal advice")


def domain_inference_example():
    """Domain-specific inference example."""
    
    print("\n" + "=" * 60)
    print("Domain-Specific Inference")
    print("=" * 60)
    
    print("\n1. Finance inference:")
    print("""
    # Load finance model
    engine = create_inference_engine(
        model_path="outputs/finance-model/best",
        tokenizer_path="tokenizer/finance_tokenizer.model",
    )
    
    # Query
    response = engine.generate(
        prompt="分析贵州茅台的财务状况",
        max_new_tokens=512,
        temperature=0.3,  # Lower temp for factual accuracy
    )
    """)
    
    print("\n2. Medical inference:")
    print("""
    # Load medical model
    engine = create_inference_engine(
        model_path="outputs/medical-model/best",
        tokenizer_path="tokenizer/medical_tokenizer.model",
    )
    
    # Query with safety checks
    response = engine.generate(
        prompt="患者症状：发热、咳嗽、乏力",
        max_new_tokens=256,
        temperature=0.2,
    )
    # Add disclaimer to output
    """)
    
    print("\n3. Legal inference:")
    print("""
    # Load legal model
    engine = create_inference_engine(
        model_path="outputs/legal-model/best",
        tokenizer_path="tokenizer/legal_tokenizer.model",
    )
    
    # Query
    response = engine.generate(
        prompt="劳动合同到期不续签如何赔偿？",
        max_new_tokens=512,
        temperature=0.3,
    )
    # Add jurisdiction and disclaimer
    """)


def multi_domain_deployment():
    """Multi-domain deployment architecture."""
    
    print("\n" + "=" * 60)
    print("Multi-Domain Deployment Architecture")
    print("=" * 60)
    
    print("""
    ┌─────────────────────────────────────────────────────┐
    │                 API Gateway                          │
    │         (Routing based on domain)                    │
    └──────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬──────────────┐
        │              │              │              │
        ▼              ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Finance │   │ Medical │   │  Legal  │   │ General │
   │  Model  │   │  Model  │   │  Model  │   │  Model  │
   │ (7B)    │   │ (7B)    │   │ (7B)    │   │ (7B)    │
   └─────────┘   └─────────┘   └─────────┘   └─────────┘
        │              │              │              │
        └──────────────┴──────────────┴──────────────┘
                       │
              ┌────────▼────────┐
              │  Load Balancer  │
              │   & Cache       │
              └─────────────────┘
    """)
    
    print("\nDomain routing logic:")
    print("""
    def route_request(query: str) -> str:
        # Domain classification
        finance_keywords = ['股票', '基金', '投资', '财报', 'GDP']
        medical_keywords = ['症状', '诊断', '治疗', '药物', '医院']
        legal_keywords = ['合同', '法律', '诉讼', '赔偿', '法规']
        
        if any(kw in query for kw in finance_keywords):
            return "finance"
        elif any(kw in query for kw in medical_keywords):
            return "medical"
        elif any(kw in query for kw in legal_keywords):
            return "legal"
        else:
            return "general"
    """)


if __name__ == "__main__":
    finance_domain_example()
    medical_domain_example()
    legal_domain_example()
    domain_inference_example()
    multi_domain_deployment()
