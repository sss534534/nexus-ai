#!/bin/bash
# Nexus-7B Serving Script

set -e

# Configuration
MODEL_PATH=${MODEL_PATH:-"checkpoints/best"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"tokenizer/nexus_tokenizer.model"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-1}

# Precision
PRECISION=${PRECISION:-"bf16"}  # fp32, fp16, bf16, int8, int4

# Quantization
QUANTIZE=${QUANTIZE:-"false"}

# Enterprise features
ENTERPRISE_CONFIG=${ENTERPRISE_CONFIG:-"configs/enterprise_config.yaml"}

echo "=========================================="
echo "Nexus-7B Model Server"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Precision: $PRECISION"
echo "Quantize: $QUANTIZE"
echo "=========================================="

# Check model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

# Check tokenizer exists
if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "Error: Tokenizer not found: $TOKENIZER_PATH"
    exit 1
fi

# Set quantization flag
QUANTIZE_FLAG=""
if [ "$QUANTIZE" == "true" ]; then
    QUANTIZE_FLAG="--quantize"
fi

# Start server
echo "Starting server..."
python -m nexus_llm.cli serve \
    --model-path $MODEL_PATH \
    --tokenizer-path $TOKENIZER_PATH \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --precision $PRECISION \
    $QUANTIZE_FLAG \
    --enterprise-config $ENTERPRISE_CONFIG \
    --log-level INFO

echo "=========================================="
echo "Server stopped"
echo "=========================================="

# Health check
echo "Testing health endpoint..."
curl -s http://localhost:$PORT/health | python -m json.tool