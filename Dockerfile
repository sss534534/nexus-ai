# Nexus-7B Docker Image
# Multi-stage build for optimized production image

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY pyproject.toml /tmp/

# Install dependencies
WORKDIR /tmp
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -e . --no-deps && \
    pip install \
        transformers \
        accelerate \
        deepspeed \
        sentencepiece \
        protobuf \
        safetensors \
        datasets \
        wandb \
        fastapi \
        uvicorn \
        prometheus-client \
        bitsandbytes \
        flash-attn --no-build-isolation

# =============================================================================
# Stage 2: Production
# =============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/model \
    TOKENIZER_PATH=/app/tokenizer \
    CONFIG_PATH=/app/configs \
    PORT=8000 \
    WORKERS=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app directory
WORKDIR /app

# Copy application code
COPY src/nexus_llm /app/nexus_llm
COPY configs /app/configs
COPY scripts /app/scripts

# Copy model files (can be mounted as volume instead)
# COPY model /app/model
# COPY tokenizer /app/tokenizer

# Create non-root user
RUN groupadd -r nexus && useradd -r -g nexus nexus && \
    chown -R nexus:nexus /app

# Switch to non-root user
USER nexus

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:$PORT/health')" || exit 1

# Default command
CMD ["python", "-m", "nexus_llm.cli", "serve", \
     "--model-path", "/app/model", \
     "--tokenizer-path", "/app/tokenizer", \
     "--host", "0.0.0.0", \
     "--port", "8000"]

# =============================================================================
# Stage 3: Development
# =============================================================================
FROM python:3.10 as development

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    vim \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install in development mode
RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

# Default command
CMD ["/bin/bash"]

# =============================================================================
# Stage 4: Training
# =============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as training

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    build-essential \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -e ".[all]" && \
    pip install flash-attn --no-build-isolation

# Default command for training
CMD ["python", "-m", "nexus_llm.cli", "train", \
     "--model-config", "configs/model_config.yaml", \
     "--training-config", "configs/training_config.yaml"]
