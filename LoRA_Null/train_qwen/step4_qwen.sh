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

# 转为绝对路径，避免 HuggingFace 把相对路径误判为 repo_id 去远端下载
MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
echo "Using model path: $MODEL_PATH"

CUDA_VISIBLE_DEVICES=0 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --output_path result_path/qwen_result.json \
    --tasks triviaqa,webqs,nq_open \
    --batch_size 64 \
    --max_batch_size 64 \
    --device cuda
