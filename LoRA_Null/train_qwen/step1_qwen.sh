#!/bin/bash
# Step 1: 构建 Qwen2.5-7B-Instruct 的 LoRA-Null 适配器
# 使用 singular_aware 模式（基于协方差矩阵零空间）

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

CUDA_VISIBLE_DEVICES=0 python train_qwen/build_adapter_qwen.py \
    --model_id "Qwen/Qwen2.5-7B-Instruct" \
    --singular_aware \
    --use_cache \
    --r 128 \
    --calib_dataset "nqopen" \
    --calib_loader_size 256 \
    --save_model \
    --save_path save_LoRA_Null_adapter_qwen25_7b_PT_128
