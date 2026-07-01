"""
基于 Qwen2.5-7B 的医学持续预训练 (CPT) — LoRA-Null V1 训练脚本

与 SFT 版本 (train_model_qwen.py) 的核心区别：
1. 数据处理：去掉 Chat 模板，使用纯文本，整个序列计算 loss（next-token prediction）
2. 参数冻结：额外解冻 embed_tokens 和 lm_head，学习医学术语
3. 数据格式：要求 {"text": "..."} 格式，而非 {query, response}

使用方式：
    python train_qwen/train_model_qwen_for_pretrain.py \
        --model_name_or_path <step1_output> \
        --data_path <medical_text_data> \
        --output_dir <output_dir> \
        --Null_mode True \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 128 \
        --learning_rate 2e-5 \
        --bf16 True
"""

import copy
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List

# 将项目根目录加入 sys.path，以便 import adapterlib（必须在其他 import 之前）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import transformers
from transformers import Trainer
from datasets import load_dataset

IGNORE_INDEX = -100


def get_nb_trainable_parameters(model) -> tuple:
    """统计可训练参数数量"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            num_bytes = param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
            num_params = num_params * 2 * num_bytes
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-7B-Instruct")
    data_path: str = field(default=None, metadata={"help": "Path to the training data (json/jsonl with 'text' field)."})
    dataset_split: str = field(
        default="train", metadata={"help": "Dataset split to use."}
    )
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    Null_mode: bool = field(default=True, metadata={"help": "True for Null mode"})
    # CPT 专用：是否解冻 embedding 和 lm_head（学习领域词汇）
    train_embeddings: bool = field(
        default=True,
        metadata={"help": "Whether to train embed_tokens and lm_head for domain vocabulary."}
    )
    optim: str = field(default="adamw_torch")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


# ========== CPT 数据处理：纯文本 + 全序列 loss ==========

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


def cpt_preprocess(
    texts: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    CPT 预处理：纯文本 tokenize，整个序列计算 loss（无标签掩码）。
    与 SFT 版 preprocess 的区别：不做 source/target 分离，labels = input_ids。
    """
    tokenized = _tokenize_fn(texts, tokenizer)
    input_ids = tokenized["input_ids"]
    # CPT 核心区别：labels = input_ids，整个序列都计算 next-token loss
    labels = copy.deepcopy(input_ids)
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForCPTDataset(object):
    """Collate examples for continual pre-training (full sequence loss)."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def cpt_tokenize_function(examples, tokenizer):
    """
    CPT tokenize 函数：纯文本处理，整个序列计算 loss。
    要求数据格式为 {"text": "..."}。
    """
    texts = examples["text"]
    data_dict = cpt_preprocess(texts, tokenizer)
    return data_dict


# ========== 训练主函数 ==========

def train():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    # ===== 加载模型 =====
    if script_args.Null_mode:
        print("=" * 60)
        print("Train in Null V1 mode for CPT (Qwen2)")
        print(f"  Model: {script_args.model_name_or_path}")
        print(f"  Data:  {script_args.data_path}")
        print(f"  Max length: {script_args.model_max_length}")
        print(f"  Train embeddings: {script_args.train_embeddings}")
        print("=" * 60)

        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
        )

        # Null 模式参数冻结逻辑
        for n, p in model.named_parameters():
            # 1. 非 ALinear/BLinear 的参数全部冻结
            if "ALinear" not in n and "BLinear" not in n and p.requires_grad:
                p.requires_grad = False
            # 2. PALinear/PBLinear 也冻结（CorDA_adapter2 中的投影矩阵）
            if ("PALinear" in n or "PBLinear" in n) and p.requires_grad:
                p.requires_grad = False

        # CPT 专用：解冻 embedding 和 lm_head，学习医学术语
        if script_args.train_embeddings:
            print("Unfreezing embed_tokens and lm_head for domain vocabulary learning...")
            for n, p in model.named_parameters():
                if "embed_tokens" in n or "lm_head" in n:
                    p.requires_grad = True
                    print(f"  Unfrozen: {n} ({p.numel():,} params)")
    else:
        raise ValueError(
            "CPT 模式当前仅支持 Null_mode=True (LoRA-Null V1)。"
            "如需全量微调，请使用 train_model_qwen.py。"
        )

    print("\n===== Model structure =====")
    print(model)

    print("\n===== Trainable parameters =====")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {n}  |  {p.numel():,} params")

    trainable_params, all_param = get_nb_trainable_parameters(model)
    print(
        f"\ntrainable params: {trainable_params:,d} || all params: {all_param:,d} "
        f"|| trainable%: {100 * trainable_params / all_param:.4f}%"
    )

    # ===== 加载 tokenizer =====
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ===== 加载并处理 CPT 数据 =====
    print(f"\nLoading CPT data from: {script_args.data_path}")

    # 支持本地 JSON/JSONL 文件和 HuggingFace Hub 数据集
    if os.path.exists(script_args.data_path):
        # 本地文件：根据扩展名选择加载方式
        ext = os.path.splitext(script_args.data_path)[1].lower()
        if ext in (".jsonl", ".json"):
            raw_train_datasets = load_dataset(
                "json", data_files=script_args.data_path, split=script_args.dataset_split
            )
        else:
            raw_train_datasets = load_dataset(
                script_args.data_path, split=script_args.dataset_split
            )
    else:
        # HuggingFace Hub 数据集
        raw_train_datasets = load_dataset(script_args.data_path, split=script_args.dataset_split)

    print(f"Dataset loaded: {len(raw_train_datasets)} samples")
    print(f"Example: {raw_train_datasets[0]}")

    train_dataset = raw_train_datasets.map(
        cpt_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=16,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on CPT dataset",
        fn_kwargs={"tokenizer": tokenizer},
    )

    data_collator = DataCollatorForCPTDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)

    # ===== 训练 =====
    trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
    model.config.use_cache = False

    print("\n===== Starting CPT training =====")
    trainer.train()

    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak memory: {peak_memory / 1024 ** 2:.2f} MB")

    # ===== 保存模型 =====
    trainer.save_state()
    model.save_pretrained(os.path.join(script_args.output_dir, 'ft'))
    tokenizer.save_pretrained(os.path.join(script_args.output_dir, 'ft'))

    # 复制 mapping 文件到 ft/ 目录，确保 auto_map 能找到
    import shutil
    mapping_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mapping')
    for fname in ['configuration_oursvd_qwen2.py', 'modeling_oursvd_qwen2.py']:
        src = os.path.join(mapping_dir, fname)
        dst = os.path.join(script_args.output_dir, 'ft', fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    print(f"\nModel saved to: {os.path.join(script_args.output_dir, 'ft')}")
    print("CPT training completed.")


if __name__ == "__main__":
    train()
