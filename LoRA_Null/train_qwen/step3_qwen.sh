#!/bin/bash
# Step 3: 合并 Qwen2.5-7B-Instruct 的 LoRA-Null 适配器

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

MODEL_PATH=$1
SAVE_PATH=$2

if [ -z "$MODEL_PATH" ] || [ -z "$SAVE_PATH" ]; then
    echo "Usage: sh train_qwen/step3_qwen.sh <model_path> <save_path>"
    echo "Example: sh train_qwen/step3_qwen.sh save_LoRA_Null_adapter_qwen25_7b_PT_128_math_Null_v1_trained/ft save_LoRA_Null_adapter_qwen25_7b_PT_128_math_Null_v1_merged"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python train_qwen/merge_adapter_for_Null_qwen.py \
    --model_id $MODEL_PATH \
    --save_path $SAVE_PATH
