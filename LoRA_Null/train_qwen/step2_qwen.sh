#!/bin/bash
# Step 2: 训练 Qwen2.5-7B-Instruct 的 LoRA-Null 适配器
# 支持 Null V1（训练 A+B）和 Null V2（冻结 A，仅训练 B）

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

BASE_MODEL=$1
OUTPUT=$2

if [ -z "$BASE_MODEL" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: sh train_qwen/step2_qwen.sh <base_model> <output_dir>"
    echo "Example: sh train_qwen/step2_qwen.sh save_LoRA_Null_adapter_qwen25_7b_PT_128 save_LoRA_Null_adapter_qwen25_7b_PT_128_math_Null_v1_trained"
    exit 1
fi

# ===== Null V1: 训练 ALinear 和 BLinear =====
echo "Training Null V1 (train ALinear + BLinear)..."
CUDA_VISIBLE_DEVICES=0 python -u train_qwen/train_model_qwen.py \
    --model_name_or_path $BASE_MODEL \
    --output_dir ${OUTPUT}_Null_v1 \
    --Null_mode True \
    --data_path meta-math/MetaMathQA \
    --dataset_split "train[:100000]" \
    --dataset_field query response \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --report_to none \
    --lora_dropout 0.0 \
    --lora_alpha 128

# ===== Null V2: 冻结 ALinear，仅训练 BLinear =====
echo "Training Null V2 (freeze ALinear, train BLinear only)..."
CUDA_VISIBLE_DEVICES=0 python -u train_qwen/train_model_qwen_freeze_a.py \
    --model_name_or_path $BASE_MODEL \
    --output_dir ${OUTPUT}_Null_v2 \
    --Null_mode True \
    --data_path meta-math/MetaMathQA \
    --dataset_split "train[:100000]" \
    --dataset_field query response \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --report_to none \
    --lora_dropout 0.0 \
    --lora_alpha 128
