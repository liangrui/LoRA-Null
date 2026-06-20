#!/bin/bash
# Step 5: 下游任务评估（GSM8K + MATH）

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

MODEL_PATH=$1

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: sh train_qwen/step5_qwen.sh <model_path>"
    echo "Example: sh train_qwen/step5_qwen.sh save_LoRA_Null_adapter_qwen25_7b_PT_128_math_Null_v1_merged"
    exit 1
fi

# GSM8K 评估
echo "Running GSM8K inference..."
CUDA_VISIBLE_DEVICES=0 python inference/gsm8k_inference.py \
    --model $MODEL_PATH \
    --batch_size 60

# MATH 评估
echo "Running MATH inference..."
CUDA_VISIBLE_DEVICES=0 python inference/MATH_inference.py \
    --model $MODEL_PATH \
    --batch_size 50
