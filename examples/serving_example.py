"""
Example: Model Serving and Deployment
"""

import json
import asyncio
from pathlib import Path

from nexus_llm.serving import (
    InferenceEngine,
    InferenceConfig,
    ModelServer,
    create_inference_engine,
    create_model_server,
)
from nexus_llm.enterprise import EnterpriseConfig, EnterpriseManager


def basic_serving_example():
    """Basic model serving example."""
    
    print("=" * 50)
    print("Basic Model Serving")
    print("=" * 50)
    
    # Inference configuration
    config = InferenceConfig(
        model_path="checkpoints/best",
        tokenizer_path="tokenizer/nexus_tokenizer.model",
        device="cuda",
        precision="bf16",
        max_new_tokens=512,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        batch_size=1,
        use_cache=True,
    )
    
    print("\nInference Configuration:")
    print(f"  Model path: {config.model_path}")
    print(f"  Precision: {config.precision}")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  Temperature: {config.temperature}")
    
    print("\nPython API usage:")
    print("""
    # Create inference engine
    engine = create_inference_engine(
        model_path="checkpoints/best",
        tokenizer_path="tokenizer/nexus_tokenizer.model",
    )
    
    # Generate text
    response = engine.generate(
        prompt="什么是人工智能？",
        max_new_tokens=256,
        temperature=0.7,
    )
    
    print(response["text"])
    """)
    
    print("\nBatch generation:")
    print("""
    # Generate for multiple prompts
    responses = engine.batch_generate(
        prompts=[
            "什么是深度学习？",
            "什么是自然语言处理？",
            "什么是Transformer？",
        ],
        max_new_tokens=128,
    )
    
    for r in responses:
        print(r["text"])
    """)


def streaming_example():
    """Streaming generation example."""
    
    print("\n" + "=" * 50)
    print("Streaming Generation")
    print("=" * 50)
    
    print("\nStreaming API usage:")
    print("""
    # Stream tokens one by one
    async def stream_example():
        for token in await engine.generate_stream(
            prompt="请写一篇关于AI的文章",
            max_new_tokens=500,
        ):
            print(token, end="", flush=True)
    
    asyncio.run(stream_example())
    """)
    
    print("\nFastAPI streaming endpoint:")
    print("""
    POST /generate/stream
    
    {
        "prompt": "请写一篇关于AI的文章",
        "max_new_tokens": 500,
        "stream": true
    }
    
    Response (SSE):
    data: {"token": "人"}
    data: {"token": "工"}
    data: {"token": "智"}
    ...
    data: [DONE]
    """)


def quantization_example():
    """Quantization for efficient deployment."""
    
    print("\n" + "=" * 50)
    print("Quantization Deployment")
    print("=" * 50)
    
    print("\nQuantization options:")
    
    quantization_configs = [
        ("BF16", "bf16", "~16GB", "~50 tokens/s"),
        ("FP16", "fp16", "~16GB", "~50 tokens/s"),
        ("INT8", "int8", "~8GB", "~60 tokens/s"),
        ("INT4", "int4", "~5GB", "~70 tokens/s"),
    ]
    
    print("\n| Precision | Memory | Speed |")
    print("|-----------|--------|-------|")
    for name, precision, memory, speed in quantization_configs:
        print(f"| {name} | {memory} | {speed} |")
    
    print("\nQuantization configuration:")
    print("""
    # INT8 quantization
    config = InferenceConfig(
        model_path="checkpoints/best",
        precision="int8",
        quantization_enabled=True,
        load_in_8bit=True,
    )
    
    # INT4 quantization
    config = InferenceConfig(
        model_path="checkpoints/best",
        precision="int4",
        quantization_enabled=True,
        load_in_4bit=True,
    )
    """)
    
    print("\nCommand line:")
    print("   nexus-serve --model-path checkpoints/best --precision int8 --quantize")


def enterprise_deployment_example():
    """Enterprise deployment example."""
    
    print("\n" + "=" * 50)
    print("Enterprise Deployment")
    print("=" * 50)
    
    # Enterprise configuration
    enterprise_config = EnterpriseConfig(
        monitoring_enabled=True,
        prometheus_port=9090,
        encryption_enabled=True,
        audit_logging=True,
        access_control=True,
        api_key_required=True,
        rate_limit_enabled=True,
        rate_limit_requests=100,
        rate_limit_window=60,
    )
    
    print("\nEnterprise Features:")
    print(f"  Monitoring: {enterprise_config.monitoring_enabled}")
    print(f"  Prometheus port: {enterprise_config.prometheus_port}")
    print(f"  Encryption: {enterprise_config.encryption_enabled}")
    print(f"  Audit logging: {enterprise_config.audit_logging}")
    print(f"  Access control: {enterprise_config.access_control}")
    print(f"  Rate limit: {enterprise_config.rate_limit_requests} requests/{enterprise_config.rate_limit_window}s")
    
    print("\nDeployment code:")
    print("""
    # Create enterprise server
    server = create_model_server(
        model_path="checkpoints/best",
        tokenizer_path="tokenizer/nexus_tokenizer.model",
        enterprise_config=enterprise_config,
    )
    
    # Start server
    server.run(host="0.0.0.0", port=8000)
    """)
    
    print("\nAPI authentication:")
    print("""
    # Generate API key
    api_key = security_manager.generate_api_key(
        user="admin",
        permissions=["read", "write"],
    )
    
    # Use API key in requests
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(
        "http://localhost:8000/generate",
        headers=headers,
        json={"prompt": "Hello"}
    )
    """)
    
    print("\nPrometheus metrics:")
    print("""
    # Available metrics
    - nexus_train_loss
    - nexus_train_steps_total
    - nexus_inference_requests_total
    - nexus_inference_latency_seconds
    - nexus_inference_tokens_total
    - nexus_gpu_memory_used_bytes
    - nexus_gpu_utilization_percent
    
    # Access metrics
    curl http://localhost:9090/metrics
    """)


def docker_deployment_example():
    """Docker deployment example."""
    
    print("\n" + "=" * 50)
    print("Docker Deployment")
    print("=" * 50)
    
    dockerfile = '''
FROM python:3.10-slim

# Install dependencies
RUN pip install torch transformers accelerate fastapi uvicorn

# Copy model
COPY checkpoints/best /app/model
COPY tokenizer /app/tokenizer

# Copy application
COPY src/nexus_llm /app/nexus_llm

# Set environment
ENV MODEL_PATH=/app/model
ENV TOKENIZER_PATH=/app/tokenizer/model

# Run server
CMD ["python", "-m", "nexus_llm.cli", "serve", \
     "--model-path", "$MODEL_PATH", \
     "--tokenizer-path", "$TOKENIZER_PATH", \
     "--port", "8000"]
'''
    
    print("\nDockerfile:")
    print(dockerfile)
    
    print("\nBuild and run:")
    print("""
    # Build image
    docker build -t nexus-7b:latest .
    
    # Run container
    docker run -d \
        --name nexus-server \
        --gpus all \
        -p 8000:8000 \
        -p 9090:9090 \
        nexus-7b:latest
    
    # Test
    curl http://localhost:8000/health
    """)


def kubernetes_deployment_example():
    """Kubernetes deployment example."""
    
    print("\n" + "=" * 50)
    print("Kubernetes Deployment")
    print("=" * 50)
    
    deployment_yaml = '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-7b
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nexus-7b
  template:
    metadata:
      labels:
        app: nexus-7b
    spec:
      containers:
      - name: nexus-server
        image: nexus-7b:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
        env:
        - name: MODEL_PATH
          value: "/app/model"
        volumeMounts:
        - name: model-storage
          mountPath: /app/model
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: nexus-model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: nexus-7b-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: nexus-7b
'''
    
    print("\nKubernetes deployment:")
    print(deployment_yaml)
    
    print("\nDeploy commands:")
    print("""
    # Create namespace
    kubectl create namespace nexus
    
    # Create PVC
    kubectl apply -f pvc.yaml
    
    # Deploy
    kubectl apply -f deployment.yaml -n nexus
    
    # Check status
    kubectl get pods -n nexus
    kubectl get services -n nexus
    
    # Access service
    kubectl port-forward svc/nexus-7b-service 8000:80 -n nexus
    """)


if __name__ == "__main__":
    basic_serving_example()
    streaming_example()
    quantization_example()
    enterprise_deployment_example()
    docker_deployment_example()
    kubernetes_deployment_example()