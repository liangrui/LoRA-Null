#!/bin/bash
# Step 4: 世界知识评估（TriviaQA, WebQS, NQ Open）

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

MODEL_PATH=$1

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: sh train_qwen/step4_qwen.sh <model_path>"
    echo "Example: sh train_qwen/step4_qwen.sh save_LoRA_Null_adapter_qwen25_7b_PT_128_math_Null_v1_merged"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH \
    --output_path result_path/qwen_result.json \
    --tasks triviaqa,webqs,nq_open \
    --batch_size 64 \
    --max_batch_size 64 \
    --device cuda
