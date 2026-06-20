# 适配 Qwen2.5-7B-Instruct 训练计划

## 摘要

将 LoRA-Null 项目从仅支持 LLaMA 系列扩展为同时支持 Qwen2.5-7B-Instruct。新增代码放在 `train_qwen/` 目录下，保持原有 LLaMA 代码不动。

## 现状分析

### LLaMA-2-7B vs Qwen2.5-7B-Instruct 架构差异

| 特性 | LLaMA-2-7B | Qwen2.5-7B-Instruct |
|------|-----------|---------------------|
| 模型类 | LlamaForCausalLM | Qwen2ForCausalLM |
| Config类 | LlamaConfig | Qwen2Config |
| hidden_size | 4096 | 3584 |
| intermediate_size | 11008 | 18944 |
| num_hidden_layers | 32 | 28 |
| num_attention_heads | 32 | 28 |
| num_key_value_heads | 32 (MHA) | 4 (GQA) |
| head_dim | 128 | 128 |
| vocab_size | 32000 | 152064 |
| max_position_embeddings | 4096 | 32768 |
| rope_theta | 10000.0 | 1000000.0 |
| rms_norm_eps | 1e-6 | 1e-6 |
| hidden_act | silu | silu |
| tie_word_embeddings | False | False |
| q_proj bias | False | **True** |
| k_proj bias | False | **True** |
| v_proj bias | False | **True** |
| o_proj bias | False | False |
| gate_proj bias | False | False |
| up_proj bias | False | False |
| down_proj bias | False | False |
| Chat模板 | LLaMA Chat (`[INST]...[/INST]`) | Qwen Chat (`<|im_start|>...<|im_end|>`) |
| tokenizer pad_token | eos_token | **已有pad_token** |
| 模型来源 | HuggingFace | ModelScope (`Qwen/Qwen2.5-7B-Instruct`) |

### 代码中 LLaMA 特定的位置

1. **mapping/modeling_oursvd_llama.py** — 类名 `CovSVDLlamaForCausalLM` 继承 `LlamaForCausalLM`
2. **mapping/configuration_oursvd_llama.py** — 类名 `CovSVDLlamaConfig` 继承 `PretrainedConfig`，model_type="llama"
3. **build_adapter.py** — auto_map 注册为 Llama 架构
4. **merge_adapter_for_Null.py** — 导入 `LlamaSdpaAttention`, `LlamaMLP`，硬编码层名列表，合并后恢复为 `LlamaForCausalLM`
5. **train_model.py / train_model_freeze_a.py** — LLaMA Chat 格式模板，tokenizer.pad_token_id 设置
6. **adapterlib/datautils.py** — LLaMA Chat 格式模板 `llama_chat_format`
7. **step1-5.sh** — 硬编码 LLaMA 模型路径

### 关键发现：适配器核心逻辑是模型无关的

- `CorDA_adapter`、`CorDA_adapter2`、`CovSVDLinear` 类本身不依赖任何 LLaMA 特定代码
- `decomposition.py` 中的 SVD 分解逻辑完全通用（操作 `nn.Linear` 层）
- `act_aware_utils.py` 中的校准逻辑完全通用（Hook 注册在 `nn.Linear` 上）
- `build_model/build_model2/build_model3` 遍历 `nn.Linear` 层的逻辑通用
- **唯一需要修改的是：模型配置类、自定义模型类、Chat 模板、合并脚本中的层名和架构恢复**

## 实施方案

### 目录结构

```
/workspace/LoRA_Null/train_qwen/
├── mapping/
│   ├── configuration_oursvd_qwen2.py    # Qwen2 自定义配置
│   └── modeling_oursvd_qwen2.py         # Qwen2 自定义模型
├── build_adapter_qwen.py                # Qwen2 适配器构建脚本
├── train_model_qwen.py                  # Qwen2 Null V1 训练脚本
├── train_model_qwen_freeze_a.py         # Qwen2 Null V2 训练脚本
├── merge_adapter_for_Null_qwen.py       # Qwen2 适配器合并脚本
├── step1_qwen.sh                        # Qwen2 五步流水线脚本
├── step2_qwen.sh
├── step3_qwen.sh
├── step4_qwen.sh
└── step5_qwen.sh
```

### 文件 1: `train_qwen/mapping/configuration_oursvd_qwen2.py`

基于 `mapping/configuration_oursvd_llama.py` 修改：
- 类名：`CovSVDQwen2Config`（继承 `PretrainedConfig`）
- `model_type = "qwen2"`
- 默认参数对齐 Qwen2.5-7B：
  - `vocab_size=152064`, `hidden_size=3584`, `intermediate_size=18944`
  - `num_hidden_layers=28`, `num_attention_heads=28`, `num_key_value_heads=4`
  - `max_position_embeddings=32768`, `rope_theta=1000000.0`
  - 新增 `attention_bias=True`（Qwen2 的 q/k/v 有 bias）
  - 新增 `lora_r` 参数
- `base_model_tp_plan` 对齐 Qwen2 的层名

### 文件 2: `train_qwen/mapping/modeling_oursvd_qwen2.py`

基于 `mapping/modeling_oursvd_llama.py` 修改：
- 导入 `Qwen2ForCausalLM` 替代 `LlamaForCausalLM`
- 类名：`CovSVDQwen2ForCausalLM`（继承 `Qwen2ForCausalLM`）
- `config_class = CovSVDQwen2Config`
- `__init__` 中替换 `nn.Linear` 为 `CovSVDLinear`（逻辑不变，只是基类变了）
- `CovSVDLinear` 类保持不变（它本身就是通用的）

### 文件 3: `train_qwen/build_adapter_qwen.py`

基于 `build_adapter.py` 修改：
- auto_map 注册改为：
  ```python
  config["auto_map"] = {
      "AutoConfig": "configuration_oursvd_qwen2.CovSVDQwen2Config",
      "AutoModelForCausalLM": "modeling_oursvd_qwen2.CovSVDQwen2ForCausalLM",
  }
  config["architectures"] = ["CovSVDQwen2ForCausalLM"]
  ```
- 复制 mapping 文件到 save_path 时改为复制 qwen2 版本
- 其他逻辑（校准、分解）完全复用 `adapterlib/` 下的通用代码

### 文件 4: `train_qwen/train_model_qwen.py`

基于 `train_model.py` 修改：
- **Chat 模板**：从 LLaMA Chat 格式改为 Qwen2 Chat 格式
  ```python
  QWEN_CHAT_TEMPLATE = (
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\n{instruction}<|im_end|>\n"
      "<|im_start|>assistant\n{response}<|im_end|>"
  )
  ```
- **Tokenizer 处理**：
  - Qwen2.5 已有 pad_token，不需要 `tokenizer.pad_token_id = tokenizer.eos_token_id`
  - 改为：`if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id`
- **PROMPT 模板**：使用 Qwen2 的 `<|im_start|>/<|im_end|>` 格式替代 Alpaca 格式
- 其余逻辑（Null_mode 冻结、LoRA 模式等）完全不变

### 文件 5: `train_qwen/train_model_qwen_freeze_a.py`

基于 `train_model_freeze_a.py` 修改，同上 Chat 模板和 Tokenizer 适配。

### 文件 6: `train_qwen/merge_adapter_for_Null_qwen.py`

基于 `merge_adapter_for_Null.py` 修改：
- 移除 `LlamaSdpaAttention` 和 `LlamaMLP` 的导入（不再需要）
- 层名匹配逻辑保持不变（q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj 在 Qwen2 中名称相同）
- 合并后配置恢复：
  ```python
  config["architectures"] = ["Qwen2ForCausalLM"]  # 替代 LlamaForCausalLM
  del config["lora_r"]
  del config["auto_map"]
  ```

### 文件 7-11: `train_qwen/step1_qwen.sh` ~ `step5_qwen.sh`

基于 `step1.sh` ~ `step5.sh` 修改：
- 模型路径改为 `Qwen/Qwen2.5-7B-Instruct`（ModelScope）
- 脚本路径指向 `train_qwen/` 目录
- 保存路径命名调整

### 不需要修改的文件（直接复用）

以下文件完全通用，`train_qwen/` 下的脚本直接 import 原始路径：
- `adapterlib/decomposition.py` — CorDA_adapter、分解逻辑
- `adapterlib/act_aware_utils.py` — 校准逻辑
- `adapterlib/datautils.py` — 数据加载（但 Chat 模板部分在训练脚本中覆盖）
- `adapterlib/evaluate_utils.py` — 评估逻辑
- `inference/` — 推理脚本通用

## 假设与决策

1. **不修改原始代码**：所有新增代码放在 `train_qwen/` 下，原始 LLaMA 代码保持不变
2. **复用 adapterlib**：核心分解和校准逻辑通过 import 复用，不复制
3. **ModelScope 下载**：Qwen2.5-7B-Instruct 从 ModelScope 下载，需添加 `modelscope` 依赖
4. **Chat 模板**：使用 Qwen2 原生的 `<|im_start|>/<|im_end|>` 格式
5. **层名兼容**：Qwen2 的注意力层和 MLP 层名称（q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj）与 LLaMA 相同，无需额外映射
6. **GQA 处理**：Qwen2 的 k_proj/v_proj 维度不同（512 vs 3584），但 SVD 分解逻辑是按层独立处理的，无需特殊处理

## 验证步骤

1. 检查 `train_qwen/` 目录结构完整性
2. 验证 Python import 路径正确（`from adapterlib.xxx import xxx`）
3. 验证 `CovSVDQwen2Config` 的默认参数与 Qwen2.5-7B-Instruct 的 config.json 一致
4. 验证 Chat 模板格式正确（`<|im_start|>system\n...<|im_end|>`）
5. 验证合并脚本恢复架构为 `Qwen2ForCausalLM`
6. 检查 step 脚本的模型路径和参数
