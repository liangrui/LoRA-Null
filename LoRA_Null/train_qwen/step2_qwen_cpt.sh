#!/bin/bash
# Step 2 (CPT): 基于 Qwen2.5-7B 的医学持续预训练 — LoRA-Null V1
#
# 与 SFT 版 step2_qwen.sh 的区别：
#   1. 训练脚本: train_model_qwen_for_pretrain.py（CPT 专用）
#   2. 数据格式: {"text": "..."} 纯文本，整个序列计算 loss
#   3. 额外解冻 embed_tokens 和 lm_head，学习医学术语
#   4. 无需 dataset_field 参数（CPT 只用 text 字段）
#
# 使用方式:
#   sh train_qwen/step2_qwen_cpt.sh <step1_output> <output_dir> <data_path>
#
# 示例:
#   sh train_qwen/step2_qwen_cpt.sh \
#       save_LoRA_Null_adapter_qwen25_7b_PT_128 \
#       save_LoRA_Null_adapter_qwen25_7b_PT_128_medical_cpt \
#       data/medical_corpus.jsonl

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

BASE_MODEL=$1
OUTPUT=$2
DATA_PATH=${3:-"data/medical_corpus.jsonl"}

if [ -z "$BASE_MODEL" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: sh train_qwen/step2_qwen_cpt.sh <base_model> <output_dir> [data_path]"
    echo ""
    echo "Arguments:"
    echo "  base_model  - Step1 输出的 CovSVD 模型路径"
    echo "  output_dir  - 训练输出目录"
    echo "  data_path   - 医学纯文本数据路径 (json/jsonl, 字段: text)"
    echo ""
    echo "Example:"
    echo "  sh train_qwen/step2_qwen_cpt.sh \\"
    echo "      save_LoRA_Null_adapter_qwen25_7b_PT_128 \\"
    echo "      save_LoRA_Null_adapter_qwen25_7b_PT_128_medical_cpt \\"
    echo "      data/medical_corpus.jsonl"
    exit 1
fi

echo "============================================================"
echo "  Medical CPT with LoRA-Null V1 (Qwen2.5-7B)"
echo "============================================================"
echo "  Base model:  $BASE_MODEL"
echo "  Output dir:  ${OUTPUT}_Null_v1"
echo "  Data path:   $DATA_PATH"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python -u train_qwen/train_model_qwen_for_pretrain.py \
    --model_name_or_path $BASE_MODEL \
    --output_dir ${OUTPUT}_Null_v1 \
    --Null_mode True \
    --train_embeddings True \
    --data_path $DATA_PATH \
    --dataset_split "train" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
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
    --model_max_length 256

echo ""
echo "CPT training completed. Output: ${OUTPUT}_Null_v1/ft"
echo "Next: sh train_qwen/step3_qwen.sh ${OUTPUT}_Null_v1/ft ${OUTPUT}_Null_v1_merged"
