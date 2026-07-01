# 医学 CPT 训练指南 (Qwen2.5-7B + LoRA-Null V1)

基于 LoRA-Null 方法，在 Qwen2.5-7B 上进行医学领域持续预训练 (Continual Pre-Training)。
在注入医学知识的同时，通过零空间投影保护通用知识不被遗忘。

---

## 目录

- [1. 方案概述](#1-方案概述)
- [2. 环境要求](#2-环境要求)
- [3. 数据准备](#3-数据准备)
- [4. 完整训练流程](#4-完整训练流程)
- [5. 评估方案](#5-评估方案)
- [6. 参数调优建议](#6-参数调优建议)
- [7. 常见问题](#7-常见问题)

---

## 1. 方案概述

### 1.1 目标

| 目标 | 说明 |
|------|------|
| 注入医学知识 | 通过医学纯文本语料进行持续预训练 |
| 保留通用知识 | 通过 LoRA-Null 零空间投影防止灾难性遗忘 |

### 1.2 原理

```
校准数据 (NQ Open 通用QA)
        ↓ 前向传播收集激活
每层 Linear 的输入激活 X
        ↓ 计算协方差 Σ = X^T·X
协方差矩阵 Σ
        ↓ SVD 分解取后 r 个特征向量
通用知识零空间 U_null
        ↓ W₀ 投影到零空间
LoRA-Null 适配器初始化 (A·X_pre ≈ 0)
        ↓ 医学语料训练
医学知识注入 (更新被限制在不影响通用知识的方向)
```

**核心优势**：医学知识与通用知识分布差异大，医学激活方向大部分落在通用知识零空间中，因此 LoRA-Null 能在保护通用知识的同时有效注入医学知识。

### 1.3 文件清单

| 文件 | 说明 |
|------|------|
| `train_qwen/step1_qwen.sh` | Step1: 构建适配器 (SVD 分解) |
| `train_qwen/train_model_qwen_for_pretrain.py` | Step2: CPT 训练主程序 |
| `train_qwen/step2_qwen_cpt.sh` | Step2: CPT 启动脚本 |
| `train_qwen/step3_qwen.sh` | Step3: 合并适配器 |
| `train_qwen/step4_qwen.sh` | Step4: 通用知识评估 |
| `train_qwen/step5_qwen.sh` | Step5: 下游任务评估 |
| `train_qwen/mapping/` | CovSVD 自定义模型代码 |

---

## 2. 环境要求

### 2.1 硬件

| 资源 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | 1×24GB (RTX 3090/4090) | 1×40GB+ (A100) |
| 内存 | 32GB | 64GB+ |
| 磁盘 | 50GB (模型+数据) | 100GB+ |

### 2.2 软件

```bash
# 核心依赖
torch >= 2.1.0
transformers >= 4.40.0
datasets >= 2.14.0
accelerate >= 0.20.0
peft >= 0.4.0

# 评估依赖
lm-eval == 0.4.0
```

### 2.3 模型下载

```bash
# 从 ModelScope 下载 Qwen2.5-7B（推荐国内用户）
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /path/to/Qwen2.5-7B-Instruct
```

---

## 3. 数据准备

### 3.1 训练数据格式

CPT 使用**纯文本格式**，每条数据包含一个 `text` 字段：

**JSONL 格式 (推荐)** — `data/medical_corpus.jsonl`：

```json
{"text": "急性心肌梗死的诊断标准包括：心肌肌钙蛋白升高超过正常参考值上限第99百分位，且伴有缺血症状、新发ST-T改变或影像学证据。"}
{"text": "糖尿病酮症酸中毒(DKA)的治疗原则包括：1.补液治疗 2.胰岛素治疗 3.纠正电解质紊乱 4.处理诱因。"}
{"text": "COPD的GOLD分级基于FEV1/FVC<0.70，根据FEV1占预计值百分比分为GOLD 1-4级。"}
```

**JSON 格式** — `data/medical_corpus.json`：

```json
[
  {"text": "急性心肌梗死的诊断标准包括..."},
  {"text": "糖尿病酮症酸中毒(DKA)的治疗原则包括..."},
  {"text": "COPD的GOLD分级基于FEV1/FVC<0.70..."}
]
```

### 3.2 数据来源建议

| 来源 | 类型 | 建议规模 | 获取方式 |
|------|------|---------|---------|
| 中文医学教材 | 纯文本 | 10万-50万条 | 内部数据 |
| PubMed 摘要 | 英文医学 | 100万+ | PubMed API |
| 中文医学百科 | 中文 | 5万-20万 | 公开数据集 |
| 临床指南 | 专业文本 | 1万-5万 | 医学网站 |
| MedQA/MedMCQA | 问答 | 1万-5万 | HuggingFace |

### 3.3 数据清洗建议

```python
# 简单的数据清洗示例
import json

cleaned = []
with open("raw_medical.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        text = item.get("text", "").strip()
        # 过滤过短或过长的文本
        if 50 < len(text) < 5000:
            cleaned.append({"text": text})

with open("medical_corpus.jsonl", "w") as f:
    for item in cleaned:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Cleaned: {len(cleaned)} samples")
```

---

## 4. 完整训练流程

### 4.1 流程总览

```
Step1: 构建适配器 (校准数据=NQ Open, 保护通用知识)
    ↓
Step2: 医学 CPT 训练 (纯文本, 全序列 loss, LoRA-Null V1)
    ↓
Step3: 合并适配器 (恢复为标准 Qwen2 架构)
    ↓
Step4: 通用知识评估 (验证知识保留)
    ↓
Step5: 下游任务评估 (验证医学能力)
```

### 4.2 Step1: 构建适配器

```bash
cd /workspace/LoRA_Null

# 使用 NQ Open 作为校准数据（保护通用知识）
# r=128, singular_aware (LoRA-Null V1)
sh train_qwen/step1_qwen.sh
```

**关键参数说明**：

| 参数 | 值 | 说明 |
|------|---|------|
| `--model_id` | `Qwen/Qwen2.5-7B-Instruct` | 基础模型 |
| `--calib_dataset` | `nqopen` | 校准数据（通用QA，保护通用知识） |
| `--singular_aware` | True | 使用 LoRA-Null V1 |
| `--r` | 128 | LoRA 秩 |
| `--calib_loader_size` | 256 | 校准样本数 |

**输出**：`save_LoRA_Null_adapter_qwen25_7b_PT_128/` (CovSVD 模型)

### 4.3 Step2: 医学 CPT 训练

```bash
# 基础用法
sh train_qwen/step2_qwen_cpt.sh \
    save_LoRA_Null_adapter_qwen25_7b_PT_128 \
    save_LoRA_Null_adapter_qwen25_7b_PT_128_medical_cpt \
    data/medical_corpus.jsonl
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 第1个参数 (base_model) | 必填 | Step1 输出的 CovSVD 模型路径 |
| 第2个参数 (output_dir) | 必填 | 训练输出目录 |
| 第3个参数 (data_path) | `data/medical_corpus.jsonl` | 医学纯文本数据路径 |

**训练参数**（在 `step2_qwen_cpt.sh` 中配置）：

| 参数 | 值 | 说明 |
|------|---|------|
| `--Null_mode` | True | LoRA-Null V1 模式 |
| `--train_embeddings` | True | 解冻 embed_tokens + lm_head，学习医学术语 |
| `--num_train_epochs` | 1 | 训练轮数 |
| `--per_device_train_batch_size` | 1 | 单卡 batch size |
| `--gradient_accumulation_steps` | 128 | 梯度累积（等效 batch=128） |
| `--learning_rate` | 2e-5 | 学习率 |
| `--model_max_length` | 512 | 最大序列长度 |
| `--bf16` | True | 使用 bfloat16 混合精度 |

**输出**：`save_LoRA_Null_adapter_qwen25_7b_PT_128_medical_cpt_Null_v1/ft/`

### 4.4 Step3: 合并适配器

```bash
sh train_qwen/step3_qwen.sh \
    save_LoRA_Null_adapter_qwen25_7b_PT_128_medical_cpt_Null_v1/ft \
    save_LoRA_Null_adapter_qwen25_7b_PT_128_medical_cpt_Null_v1_merged
```

**输出**：`save_LoRA_Null_adapter_qwen25_7b_PT_128_medical_cpt_Null_v1_merged/` (标准 Qwen2 模型)

### 4.5 一键执行

将以下内容保存为 `run_medical_cpt.sh`：

```bash
#!/bin/bash
set -e

# 配置
BASE_MODEL_NAME="save_LoRA_Null_adapter_qwen25_7b_PT_128"
CPT_OUTPUT="save_LoRA_Null_adapter_qwen25_7b_PT_128_medical_cpt"
DATA_PATH="data/medical_corpus.jsonl"
MERGED="${CPT_OUTPUT}_Null_v1_merged"

echo "===== Step1: Build adapter ====="
sh train_qwen/step1_qwen.sh

echo "===== Step2: Medical CPT training ====="
sh train_qwen/step2_qwen_cpt.sh $BASE_MODEL_NAME $CPT_OUTPUT $DATA_PATH

echo "===== Step3: Merge adapter ====="
sh train_qwen/step3_qwen.sh ${CPT_OUTPUT}_Null_v1/ft $MERGED

echo "===== Step4: Evaluate general knowledge ====="
sh train_qwen/step4_qwen.sh $MERGED

echo "===== All done! ====="
echo "Merged model: $MERGED"
```

```bash
chmod +x run_medical_cpt.sh
sh run_medical_cpt.sh
```

---

## 5. 评估方案

### 5.1 通用知识保留评估

```bash
sh train_qwen/step4_qwen.sh \
    save_LoRA_Null_adapter_qwen25_7b_PT_128_medical_cpt_Null_v1_merged
```

评估任务：

| 任务 | 数据集 | 指标 | 期望结果 |
|------|--------|------|---------|
| TriviaQA | 通用问答 | EM | 接近原始模型 |
| NQ Open | 自然问题 | EM | 接近原始模型 |
| WebQS | 网络问答 | EM | 接近原始模型 |

**判断标准**：通用知识保留率应 > 85%（相比原始模型下降 < 15%）。

### 5.2 医学知识能力评估

需要自行准备医学评估数据集。推荐：

| 评估集 | 类型 | 获取方式 |
|--------|------|---------|
| MedQA | 美国执业医师考试 | HuggingFace: bigbio/med_qa |
| MedMCQA | 印度医学入学考试 | HuggingFace: openlifescienceai/medmcqa |
| PubMedQA | 生物医学问答 | HuggingFace: bigbio/pubmed_qa |
| CMB | 中文医学综合 | HuggingFace: mbzuai/CMB |

评估示例（以 CMB 为例）：

```bash
# 使用 lm-eval-harness 评估
CUDA_VISIBLE_DEVICES=0 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=save_LoRA_Null_adapter_qwen25_7b_PT_128_medical_cpt_Null_v1_merged,trust_remote_code=True \
    --tasks cmb \
    --batch_size 32 \
    --output_path result_path/medical_eval.json
```

### 5.3 对比评估

建议同时评估以下模型作为对比基线：

| 模型 | 作用 |
|------|------|
| 原始 Qwen2.5-7B-Instruct | 上界：通用知识基准 |
| 标准 LoRA CPT | 对比：LoRA-Null 的知识保留优势 |
| 全量微调 CPT | 下界：灾难性遗忘最严重 |

---

## 6. 参数调优建议

### 6.1 关键参数

| 参数 | 调优方向 | 影响 |
|------|---------|------|
| `--r` (秩) | 增大 → 医学知识学习更强，但通用知识保留下降 | r=128 推荐 |
| `--learning_rate` | 增大 → 学习更快，但遗忘风险增加 | 2e-5 推荐 |
| `--num_train_epochs` | 增大 → 医学知识更深入，但遗忘增加 | 1-3 epochs |
| `--train_embeddings` | True → 学习医学术语；False → 纯 LoRA | 医学场景推荐 True |
| `--model_max_length` | 增大 → 长文本学习更好，显存增加 | 512-2048 |

### 6.2 场景化配置

**场景 A：医学知识注入为主，通用知识可小幅下降**

```bash
--r 256
--learning_rate 5e-5
--num_train_epochs 3
--train_embeddings True
```

**场景 B：通用知识保护为主，医学知识轻度增强**

```bash
--r 64
--learning_rate 1e-5
--num_train_epochs 1
--train_embeddings False
```

**场景 C：平衡（推荐默认）**

```bash
--r 128
--learning_rate 2e-5
--num_train_epochs 1
--train_embeddings True
```

### 6.3 校准数据选择

| 校准数据 | 保护的知识类型 | 适用场景 |
|---------|--------------|---------|
| `nqopen` (推荐) | 通用百科知识 | 医学 CPT 默认选择 |
| `wikitext2` | 维基百科文本 | 保护百科知识 |
| `c4` | 通用网页文本 | 保护通用语言能力 |
| `traivia_qa` | 冷知识问答 | 保护事实知识 |

**医学 CPT 必须用通用数据做校准**，不能用医学数据，否则零空间会保护医学知识而非通用知识。

---

## 7. 常见问题

### Q1: 训练显存不足 (OOM)

**解决方案**：

```bash
# 方案1: 减小序列长度
--model_max_length 256

# 方案2: 使用梯度检查点（在 step2_qwen_cpt.sh 中添加）
--gradient_checkpointing True

# 方案3: 关闭 embedding 训练
--train_embeddings False
```

### Q2: Step3 合并报错 "modeling_oursvd_qwen2.py not found"

**原因**：Step2 保存的 ft/ 目录缺少 mapping 文件。

**解决**：

```bash
cp train_qwen/mapping/configuration_oursvd_qwen2.py \
   train_qwen/mapping/modeling_oursvd_qwen2.py \
   save_LoRA_Null_adapter_qwen25_7b_PT_128_medical_cpt_Null_v1/ft/
```

### Q3: 通用知识遗忘严重

**排查步骤**：

1. 确认 Step1 校准数据用的是 `nqopen`（通用数据），而非医学数据
2. 降低学习率：`--learning_rate 1e-5`
3. 减小秩：`--r 64`
4. 减少 epoch：`--num_train_epochs 1`
5. 关闭 embedding 训练：`--train_embeddings False`

### Q4: 医学知识学习效果差

**排查步骤**：

1. 增大秩：`--r 256`
2. 增加训练轮数：`--num_train_epochs 3`
3. 增大学习率：`--learning_rate 5e-5`
4. 确保训练数据质量（长度适中、内容专业）
5. 增大序列长度：`--model_max_length 1024`

### Q5: 自定义模型路径

如果 Qwen2.5-7B 下载在本地，修改 `step1_qwen.sh` 中的 `--model_id`：

```bash
--model_id /path/to/Qwen2.5-7B-Instruct
```

### Q6: 使用多 GPU 训练

修改 `step2_qwen_cpt.sh` 中的启动命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    train_qwen/train_model_qwen_for_pretrain.py \
    ...
```

---

## 附录：训练数据示例

以下是一个小规模的医学 CPT 数据样例，可直接用于测试：

```json
{"text": "高血压的诊断标准：在未使用降压药物的情况下，非同日3次测量血压，收缩压≥140mmHg和/或舒张压≥90mmHg。"}
{"text": "心力衰竭的NYHA分级：I级-体力活动不受限；II级-体力活动轻度受限；III级-体力活动明显受限；IV级-休息时即出现症状。"}
{"text": "急性胰腺炎的诊断标准：满足以下3项中的2项：1.持续性上腹疼痛；2.血清淀粉酶或脂肪酶≥正常上限3倍；3.影像学检查符合胰腺炎表现。"}
{"text": "慢性肾脏病(CKD)的KDIGO分期：基于eGFR分为G1-G5期。G1:≥90; G2:60-89; G3a:45-59; G3b:30-44; G4:15-29; G5:<15 mL/min/1.73m²。"}
{"text": "肺血栓栓塞症(PTE)的临床表现：呼吸困难(最常见)、胸痛、咯血、咳嗽、晕厥。三联征(呼吸困难、胸痛、咯血)仅见于约20%的患者。"}
```

将以上内容保存为 `data/medical_corpus.jsonl` 即可用于测试训练流程。
