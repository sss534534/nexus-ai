#!/bin/bash
# Nexus-7B Training Script

set -e

# Configuration
MODEL_CONFIG="configs/model_config.yaml"
TRAINING_CONFIG="configs/training_config.yaml"
DATA_CONFIG="configs/data_config.yaml"
OUTPUT_DIR="outputs/nexus-7b-$(date +%Y%m%d-%H%M%S)"

# Number of GPUs
NUM_GPUS=${NUM_GPUS:-8}

# Training mode
MODE=${MODE:-"distributed"}  # single, distributed, lora

echo "=========================================="
echo "Nexus-7B Training"
echo "=========================================="
echo "Mode: $MODE"
echo "GPUs: $NUM_GPUS"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Copy configs
cp $MODEL_CONFIG $OUTPUT_DIR/
cp $TRAINING_CONFIG $OUTPUT_DIR/
cp $DATA_CONFIG $OUTPUT_DIR/

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

# Run training
if [ "$MODE" == "single" ]; then
    echo "Starting single-GPU training..."
    python -m nexus_llm.cli train \
        --model-config $MODEL_CONFIG \
        --training-config $TRAINING_CONFIG \
        --data-config $DATA_CONFIG \
        --output-dir $OUTPUT_DIR \
        --log-level INFO

elif [ "$MODE" == "distributed" ]; then
    echo "Starting distributed training with $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        -m nexus_llm.cli train \
        --model-config $MODEL_CONFIG \
        --training-config $TRAINING_CONFIG \
        --data-config $DATA_CONFIG \
        --output-dir $OUTPUT_DIR \
        --distributed \
        --log-level INFO

elif [ "$MODE" == "lora" ]; then
    echo "Starting LoRA fine-tuning..."
    python -m nexus_llm.cli train \
        --model-config $MODEL_CONFIG \
        --training-config configs/lora_config.yaml \
        --data-config $DATA_CONFIG \
        --output-dir $OUTPUT_DIR \
        --log-level INFO
fi

echo "=========================================="
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="

# Save training log
cp logs/latest.log $OUTPUT_DIR/training.log