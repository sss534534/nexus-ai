# Nexus-LLM 部署指南

> 本文档详细介绍 Nexus-LLM 推理服务的各种部署方案，包括单机部署、Docker 部署、多 GPU 部署、Kubernetes 部署、反向代理配置、监控告警、SSL/TLS 配置以及性能调优和高可用方案。

---

## 目录

- [单机部署](#单机部署)
- [Docker 部署](#docker-部署)
  - [使用 Docker 镜像](#使用-docker-镜像)
  - [使用 docker-compose](#使用-docker-compose)
- [多 GPU 部署](#多-gpu-部署)
- [Kubernetes 部署](#kubernetes-部署)
  - [基础部署清单](#基础部署清单)
  - [HPA 自动扩缩容](#hpa-自动扩缩容)
- [Nginx 反向代理](#nginx-反向代理)
- [Prometheus + Grafana 监控](#prometheus--grafana-监控)
  - [Prometheus 配置](#prometheus-配置)
  - [Grafana 配置](#grafana-配置)
  - [告警规则](#告警规则)
- [SSL/TLS 配置](#ssltls-配置)
- [性能调优](#性能调优)
- [高可用部署](#高可用部署)

---

## 单机部署

单机部署是最简单的部署方式，适用于开发测试和小规模生产环境。

### 环境准备

```bash
# 1. 创建虚拟环境
python -m venv nexus-env
source nexus-env/bin/activate

# 2. 安装推理服务依赖
pip install "nexus-llm[serve]"

# 3. 下载模型
huggingface-cli download nexus-ai/nexus-7b --local-dir ./models/nexus-7b
```

### 启动服务

```bash
# 前台启动（开发调试）
nexus-serve \
  --model-path ./models/nexus-7b \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096

# 后台启动（生产环境）
nohup nexus-serve \
  --model-path ./models/nexus-7b \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  > /var/log/nexus-serve.log 2>&1 &

# 使用 systemd 管理（推荐）
```

### Systemd 服务配置

创建 `/etc/systemd/system/nexus-serve.service`：

```ini
[Unit]
Description=Nexus-LLM Inference Server
After=network.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=nexus
Group=nexus
WorkingDirectory=/opt/nexus-llm
Environment="PATH=/opt/nexus-llm/venv/bin"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/opt/nexus-llm/venv/bin/nexus-serve \
  --model-path /opt/nexus-llm/models/nexus-7b \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
Restart=always
RestartSec=10
StandardOutput=append:/var/log/nexus-serve/stdout.log
StandardError=append:/var/log/nexus-serve/stderr.log

# 资源限制
LimitNOFILE=65536
MemoryMax=64G

[Install]
WantedBy=multi-user.target
```

```bash
# 启用并启动服务
sudo systemctl daemon-reload
sudo systemctl enable nexus-serve
sudo systemctl start nexus-serve

# 查看状态
sudo systemctl status nexus-serve

# 查看日志
sudo journalctl -u nexus-serve -f
```

### 验证部署

```bash
# 健康检查
curl http://localhost:8000/health

# 测试推理
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nexus-7b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50
  }'
```

---

## Docker 部署

### 使用 Docker 镜像

```bash
# 拉取镜像
docker pull nexusai/nexus-llm:latest

# 运行推理服务
docker run -d \
  --name nexus-serve \
  --gpus all \
  --shm-size=16g \
  -p 8000:8000 \
  -v ~/.nexus/models:/root/.nexus/models \
  -e CUDA_VISIBLE_DEVICES=0 \
  nexusai/nexus-llm:latest \
  nexus-serve \
    --model-path /root/.nexus/models/nexus-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9

# 查看日志
docker logs -f nexus-serve

# 停止容器
docker stop nexus-serve
```

### 使用 docker-compose

创建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  nexus-serve:
    image: nexusai/nexus-llm:latest
    container_name: nexus-serve
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./models:/root/.nexus/models
      - ./logs:/var/log/nexus
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NEXUS_MODEL_PATH=/root/.nexus/models/nexus-7b
      - NEXUS_GPU_MEMORY_UTILIZATION=0.9
      - NEXUS_MAX_MODEL_LEN=8192
      - NEXUS_MAX_NUM_SEQS=128
    command: >
      nexus-serve
        --model-path /root/.nexus/models/nexus-7b
        --host 0.0.0.0
        --port 8000
        --gpu-memory-utilization 0.9
        --max-model-len 8192
        --max-num-seqs 128
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "5"

  prometheus:
    image: prom/prometheus:latest
    container_name: nexus-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:latest
    container_name: nexus-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=nexus_admin
      - GF_USERS_ALLOW_SIGN_UP=false

volumes:
  prometheus-data:
  grafana-data:
```

```bash
# 启动所有服务
docker-compose up -d

# 查看状态
docker-compose ps
docker-compose logs -f nexus-serve

# 停止所有服务
docker-compose down

# 重新构建并启动
docker-compose up -d --build
```

---

## 多 GPU 部署

对于大模型（如 70B），单张 GPU 无法容纳完整模型，需要使用张量并行将模型分片到多张 GPU。

### 张量并行部署

```bash
# 2 GPU 张量并行（适用于 13B-30B 模型）
nexus-serve \
  --model-path ./models/nexus-30b \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --port 8000

# 4 GPU 张量并行（适用于 65B-70B 模型）
nexus-serve \
  --model-path ./models/nexus-70b \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --port 8000

# 8 GPU 张量并行（适用于超大模型）
nexus-serve \
  --model-path ./models/nexus-120b \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.92 \
  --port 8000

# 指定 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 nexus-serve \
  --model-path ./models/nexus-70b \
  --tensor-parallel-size 4 \
  --port 8000
```

### 流水线并行部署

```bash
# 2 阶段流水线并行
nexus-serve \
  --model-path ./models/nexus-70b \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 2 \
  --port 8000
```

### 多实例负载均衡

使用多实例 + Nginx 实现负载均衡：

```bash
# 实例 1 - GPU 0,1
CUDA_VISIBLE_DEVICES=0,1 nexus-serve \
  --model-path ./models/nexus-7b \
  --port 8001 &

# 实例 2 - GPU 2,3
CUDA_VISIBLE_DEVICES=2,3 nexus-serve \
  --model-path ./models/nexus-7b \
  --port 8002 &

# 实例 3 - GPU 4,5
CUDA_VISIBLE_DEVICES=4,5 nexus-serve \
  --model-path ./models/nexus-7b \
  --port 8003 &

# 实例 4 - GPU 6,7
CUDA_VISIBLE_DEVICES=6,7 nexus-serve \
  --model-path ./models/nexus-7b \
  --port 8004 &
```

---

## Kubernetes 部署

### 基础部署清单

**nexus-deployment.yaml：**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-llm
  namespace: nexus
  labels:
    app: nexus-llm
    model: nexus-7b
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nexus-llm
  strategy:
    type: Recreate  # GPU 资源独占，使用 Recreate 策略
  template:
    metadata:
      labels:
        app: nexus-llm
        model: nexus-7b
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      nodeSelector:
        gpu-type: nvidia-a100  # 调度到指定 GPU 节点
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: nexus-serve
          image: nexusai/nexus-llm:latest
          ports:
            - containerPort: 8000
              protocol: TCP
          args:
            - nexus-serve
            - --model-path
            - /models/nexus-7b
            - --host
            - 0.0.0.0
            - --port
            - "8000"
            - --gpu-memory-utilization
            - "0.9"
            - --max-model-len
            - "8192"
            - --max-num-seqs
            - "128"
          env:
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
            - name: NEXUS_LOG_LEVEL
              value: "info"
          resources:
            limits:
              nvidia.com/gpu: "1"
              memory: "32Gi"
              cpu: "8"
            requests:
              nvidia.com/gpu: "1"
              memory: "16Gi"
              cpu: "4"
          volumeMounts:
            - name: model-volume
              mountPath: /models
            - name: log-volume
              mountPath: /var/log/nexus
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120  # 模型加载需要时间
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 180
            periodSeconds: 60
            timeoutSeconds: 10
            failureThreshold: 5
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 10
            failureThreshold: 30  # 最多等待 5 分钟启动
      volumes:
        - name: model-volume
          persistentVolumeClaim:
            claimName: nexus-model-pvc
        - name: log-volume
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: nexus-llm-service
  namespace: nexus
spec:
  selector:
    app: nexus-llm
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nexus-model-pvc
  namespace: nexus
spec:
  accessModes:
    - ReadOnlyMany
  storageClassName: nfs
  resources:
    requests:
      storage: 100Gi
```

**多 GPU 张量并行部署：**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-llm-tp4
  namespace: nexus
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: nexus-serve
          image: nexusai/nexus-llm:latest
          args:
            - nexus-serve
            - --model-path
            - /models/nexus-70b
            - --tensor-parallel-size
            - "4"
            - --gpu-memory-utilization
            - "0.92"
            - --port
            - "8000"
          resources:
            limits:
              nvidia.com/gpu: "4"  # 请求 4 张 GPU
              memory: "128Gi"
```

### HPA 自动扩缩容

> 注意：由于 GPU 资源的特殊性，HPA 需要配合 Cluster Autoscaler 使用。

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nexus-llm-hpa
  namespace: nexus
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nexus-llm
  minReplicas: 1
  maxReplicas: 4
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: nexus_avg_latency_ms
        target:
          type: AverageValue
          averageValue: "100"  # 平均延迟超过 100ms 时扩容
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120  # 每 2 分钟最多扩容 1 个 Pod
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 1
          periodSeconds: 300  # 每 5 分钟最多缩容 1 个 Pod
```

---

## Nginx 反向代理

### 基础配置

```nginx
upstream nexus_backend {
    # 多实例负载均衡
    server 127.0.0.1:8001 weight=1;
    server 127.0.0.1:8002 weight=1;
    server 127.0.0.1:8003 weight=1;
    server 127.0.0.1:8004 weight=1;

    # 健康检查
    keepalive 32;
}

server {
    listen 80;
    server_name api.example.com;

    # 请求体大小限制（适用于长文本）
    client_max_body_size 10M;

    # 超时配置
    proxy_connect_timeout 10s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;

    # 流式响应支持
    proxy_buffering off;
    proxy_cache off;

    location / {
        proxy_pass http://nexus_backend;
        proxy_http_version 1.1;

        # 请求头传递
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE 流式支持
        proxy_set_header Connection '';
        proxy_set_header Accept-Encoding none;

        # 禁用缓冲（流式传输必需）
        chunked_transfer_encoding on;
        tcp_nodelay on;
    }

    # 健康检查端点（不走负载均衡）
    location /health {
        proxy_pass http://127.0.0.1:8001/health;
    }

    # Prometheus 指标（仅内网访问）
    location /metrics {
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        deny all;
        proxy_pass http://127.0.0.1:8001/metrics;
    }

    # 访问日志
    access_log /var/log/nginx/nexus_access.log;
    error_log /var/log/nginx/nexus_error.log;
}
```

### 限流配置

```nginx
# 在 http 块中定义限流区域
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=30r/m;
limit_req_zone $binary_remote_addr zone=chat_limit:10m rate=10r/m;

server {
    # ...

    # 全局限流
    limit_req zone=api_limit burst=20 nodelay;

    # Chat Completions 更严格的限流
    location /v1/chat/completions {
        limit_req zone=chat_limit burst=5 nodelay;
        proxy_pass http://nexus_backend;
        # ... 其他配置
    }

    # 返回 429 时的自定义响应
    limit_req_status 429;
}
```

---

## Prometheus + Grafana 监控

### Prometheus 配置

创建 `monitoring/prometheus.yml`：

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - alerts.yml

scrape_configs:
  - job_name: 'nexus-llm'
    static_configs:
      - targets:
          - 'nexus-serve:8000'
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'nvidia-gpu'
    static_configs:
      - targets:
          - 'node-exporter:9100'
    scrape_interval: 15s
```

**告警规则 `monitoring/alerts.yml`：**

```yaml
groups:
  - name: nexus_llm_alerts
    rules:
      - alert: NexusHighLatency
        expr: nexus_request_latency_seconds_p99 > 5.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Nexus-LLM P99 延迟过高"
          description: "P99 延迟为 {{ $value }}s，超过 5s 阈值"

      - alert: NexusHighErrorRate
        expr: rate(nexus_request_errors_total[5m]) / rate(nexus_request_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Nexus-LLM 错误率过高"
          description: "错误率为 {{ $value | humanizePercentage }}"

      - alert: NexusGPUMemoryHigh
        expr: nexus_gpu_memory_used_bytes / nexus_gpu_memory_total_bytes > 0.95
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "GPU 显存使用率过高"
          description: "GPU 显存使用率为 {{ $value | humanizePercentage }}"

      - alert: NexusServiceDown
        expr: up{job="nexus-llm"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Nexus-LLM 服务不可用"
          description: "服务已停止响应超过 2 分钟"
```

### Grafana 配置

**数据源配置 `monitoring/grafana/provisioning/datasources/prometheus.yml`：**

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

**仪表板配置 `monitoring/grafana/provisioning/dashboards/nexus.yml`：**

```yaml
apiVersion: 1
providers:
  - name: Nexus-LLM
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: false
```

### 关键监控指标

| 指标名称 | 类型 | 说明 |
|---------|------|------|
| `nexus_request_total` | Counter | 请求总数 |
| `nexus_request_latency_seconds` | Histogram | 请求延迟分布 |
| `nexus_request_errors_total` | Counter | 错误请求数 |
| `nexus_tokens_generated_total` | Counter | 生成的 token 总数 |
| `nexus_active_requests` | Gauge | 当前活跃请求数 |
| `nexus_gpu_memory_used_bytes` | Gauge | GPU 显存使用量 |
| `nexus_gpu_memory_total_bytes` | Gauge | GPU 显存总量 |
| `nexus_gpu_utilization` | Gauge | GPU 利用率 |
| `nexus_kv_cache_usage_ratio` | Gauge | KV Cache 使用率 |
| `nexus_num_seqs_running` | Gauge | 正在运行的序列数 |
| `nexus_num_seqs_waiting` | Gauge | 等待中的序列数 |

---

## SSL/TLS 配置

### 使用 Let's Encrypt

```bash
# 安装 certbot
sudo apt install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d api.example.com

# 自动续期（certbot 会自动设置 cron）
sudo certbot renew --dry-run
```

### Nginx SSL 配置

```nginx
server {
    listen 443 ssl http2;
    server_name api.example.com;

    # SSL 证书
    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    # SSL 安全配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;

    # SSL 会话缓存
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_session_tickets false;

    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;

    location / {
        proxy_pass http://nexus_backend;
        # ... 其他配置同上
    }
}

# HTTP -> HTTPS 重定向
server {
    listen 80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}
```

### Nexus-LLM 原生 SSL

```bash
# 直接在 nexus-serve 上启用 SSL
nexus-serve \
  --model-path ./models/nexus-7b \
  --ssl-certfile /etc/letsencrypt/live/api.example.com/fullchain.pem \
  --ssl-keyfile /etc/letsencrypt/live/api.example.com/privkey.pem \
  --port 8443
```

---

## 性能调优

### GPU 显存优化

```bash
# 1. 调整显存利用率（默认 0.9）
nexus-serve --gpu-memory-utilization 0.95

# 2. 使用量化模型（减少显存占用）
nexus-serve \
  --model-path ./models/nexus-7b-awq \
  --quantization awq

# 3. 减小 KV Cache（减少并发但节省显存）
nexus-serve --max-model-len 2048 --max-num-seqs 64

# 4. 启用 KV Cache 页大小调优
nexus-serve --block-size 16  # 默认 16，减小可减少碎片
```

### 吞吐量优化

```bash
# 1. 增加最大并发序列数
nexus-serve --max-num-seqs 256

# 2. 启用 CUDA Graph（减少 kernel launch 开销）
nexus-serve --enable-cuda-graph

# 3. 调整批处理参数
nexus-serve --max-num-batched-tokens 32768

# 4. 使用 FP8 量化（在支持 FP8 的 GPU 上）
nexus-serve --dtype float8
```

### 延迟优化

```bash
# 1. 预热模型（首次请求前）
nexus-serve --enforce-eager  # 禁用 CUDA Graph 以减少首次延迟

# 2. 减小最大序列长度（减少 KV Cache 查找开销）
nexus-serve --max-model-len 2048

# 3. 使用 SWA（Sliding Window Attention）加速长序列
nexus-serve --sliding-window 2048
```

### 系统级优化

```bash
# 1. 设置 GPU 持久化模式
sudo nvidia-smi -pm 1

# 2. 设置 GPU 时钟频率（最大化性能）
sudo nvidia-smi -lgc 2100,2100  # 根据具体 GPU 型号调整

# 3. 增加共享内存（Docker 部署时重要）
docker run --shm-size=32g ...

# 4. CPU 绑核（减少上下文切换）
taskset -c 0-15 nexus-serve ...

# 5. 调整网络参数
sudo sysctl -w net.core.somaxconn=65535
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=65535
sudo sysctl -w net.ipv4.tcp_tw_reuse=1
```

### 性能基准测试

```bash
# 使用 nexus-bench 工具
pip install nexus-bench

# 吞吐量测试
nexus-bench throughput \
  --url http://localhost:8000/v1/completions \
  --model nexus-7b \
  --num-prompts 1000 \
  --concurrency 32

# 延迟测试
nexus-bench latency \
  --url http://localhost:8000/v1/chat/completions \
  --model nexus-7b \
  --num-prompts 100 \
  --input-length 128 \
  --output-length 256
```

---

## 高可用部署

### 架构概览

```
                    +------------------+
                    |    DNS (Round    |
                    |    Robin)        |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     +--------+--------+          +--------+--------+
     |   Nginx (主)    |          |   Nginx (备)    |
     |   Keepalived    |          |   Keepalived    |
     |   VIP: 10.0.0.1 |          |   VIP: 10.0.0.1 |
     +--------+--------+          +--------+--------+
              |                             |
              +--------------+--------------+
                             |
              +--------------+--------------+
              |              |              |
     +--------+---+  +-------+----+  +------+------+
     |  Nexus Pod |  | Nexus Pod  |  | Nexus Pod   |
     |  GPU 0,1   |  | GPU 2,3    |  | GPU 4,5     |
     +------------+  +------------+  +-------------+
              |              |              |
              +--------------+--------------+
                             |
                    +--------+--------+
                    |   Redis (会话   |
                    |   缓存/限流)    |
                    +-----------------+
```

### Keepalived 配置

**主节点 `/etc/keepalived/keepalived.conf`：**

```conf
vrrp_instance VI_1 {
    state MASTER
    interface eth0
    virtual_router_id 51
    priority 100
    advert_int 1
    authentication {
        auth_type PASS
        auth_pass nexus_ha
    }
    virtual_ipaddress {
        10.0.0.1/24
    }
    track_script {
        chk_nexus
    }
}

vrrp_script chk_nexus {
    script "/usr/local/bin/check_nexus.sh"
    interval 5
    weight -20
    fall 3
    rise 2
}
```

**健康检查脚本 `/usr/local/bin/check_nexus.sh`：**

```bash
#!/bin/bash
response=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
if [ "$response" != "200" ]; then
    exit 1
fi
exit 0
```

### 多区域部署

```yaml
# Kubernetes 多区域部署示例
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-llm-zone-a
  namespace: nexus
spec:
  replicas: 2
  template:
    spec:
      nodeSelector:
        topology.kubernetes.io/zone: zone-a
      # ... 其他配置
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-llm-zone-b
  namespace: nexus
spec:
  replicas: 2
  template:
    spec:
      nodeSelector:
        topology.kubernetes.io/zone: zone-b
      # ... 其他配置
---
apiVersion: v1
kind: Service
metadata:
  name: nexus-llm-lb
  namespace: nexus
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
spec:
  selector:
    app: nexus-llm
  ports:
    - port: 443
      targetPort: 8000
  type: LoadBalancer
```

### 灾备与回滚

```bash
# 模型版本管理
ls /models/
# nexus-7b-v1/  nexus-7b-v2/  nexus-7b-v3/

# 快速切换模型版本（软链接）
ln -sfn /models/nexus-7b-v3 /models/nexus-7b-current

# 回滚到上一版本
ln -sfn /models/nexus-7b-v2 /models/nexus-7b-current

# Docker 镜像回滚
docker-compose down
docker tag nexusai/nexus-llm:v2.1 nexusai/nexus-llm:latest
docker-compose up -d
```

---

> 更多信息请参阅 [快速入门](./getting_started.md) 和 [架构设计](./architecture.md)。
