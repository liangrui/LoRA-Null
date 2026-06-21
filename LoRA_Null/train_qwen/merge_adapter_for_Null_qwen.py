import numpy as np
import argparse
import os
import sys

# 将项目根目录加入 sys.path（必须在其他 import 之前）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_qwen.mapping.modeling_oursvd_qwen2 import CovSVDLinear


def main(args):
    # Load model
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    print("\n---- model before merge ---\n")

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    ## merge =======
    print("\nbegin merge. \n")
    module_dict = {module: name for name, module in model.named_modules()}
    for module in module_dict.keys():
        name = module_dict[module]
        if type(module).__name__ == "CovSVDLinear":
            info = linear_info[module]
            in_features = module.BLinear.in_features
            out_features = module.ALinear.out_features
            has_bias = module.ALinear.bias is not None
            new_linear = nn.Linear(in_features, out_features, bias=has_bias)
            merged_weight = module.ALinear.weight.data @ module.BLinear.weight.data + module.weight_residual
            new_linear.weight.data = merged_weight
            if has_bias:
                new_linear.bias.data = module.ALinear.bias.data.clone()
            delattr(info["father"], info["name"])
            setattr(info["father"], info["name"], new_linear)

    print("\n---- model after merge ---\n")
    print(model)

    ## save as hugging face model
    if args.save_model:
        assert args.save_path is not None
        save_path = args.save_path

        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        config = model.config.to_dict()
        # 恢复为 Qwen2ForCausalLM 架构
        config["architectures"] = ["Qwen2ForCausalLM"]
        del config["lora_r"]
        del config["auto_map"]
        if "_name_or_path" in config:
            del config["_name_or_path"]
        import json

        json.dump(config, open(save_path + "/config.json", "w"), indent=2)

        print(f"Done merging adapter into the original model architecture in {save_path}")
        del model
        del tokenizer
    # finished


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Pretrained model ID (path to the ft/ directory of trained model)",
    )
    parser.add_argument(
        "--save_model",
        default=True,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    main(args)
