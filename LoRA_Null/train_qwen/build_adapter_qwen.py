import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adapterlib.datautils import get_calib_data
from adapterlib.act_aware_utils import calib_input_distribution, calib_fisher_info, calib_cov_distribution
from adapterlib.decomposition import build_model2
import numpy as np
import os
import sys

# 将项目根目录加入 sys.path，以便 import adapterlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main(args):
    # setting random seed of numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load model
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    # collect calibration data
    calib_loader = get_calib_data(args.calib_dataset, tokenizer, model_id, args.calib_loader_size, seed=args.seed)

    # collect covariance for CO-SVD or activation for ASVD
    if args.act_aware:
        print('Collect activation-aware data for ASVD ...')
        if "fisher" in args.scaling_method:
            calib_fisher_info(model, calib_loader, args.use_cache)
        if "abs" in args.scaling_method:
            calib_input_distribution(
                model, calib_loader, args.scaling_method, args.use_cache
            )
    elif args.cov_aware:
        print('Collecting covariance data for CovSVD ...')
        calib_cov_distribution(
            model, calib_loader, args.use_cache, args.calib_dataset, args.calib_loader_size, seed=args.seed
        )
    elif args.singular_aware:
        print('Collecting covariance data for Singular_aware ...')
        calib_cov_distribution(
            model, calib_loader, args.use_cache, args.calib_dataset, args.calib_loader_size, seed=args.seed
        )
    elif args.singular_aware_2:
        print('Collecting covariance data for Singular_aware ...')
        calib_cov_distribution(
            model, calib_loader, args.use_cache, args.calib_dataset, args.calib_loader_size, seed=args.seed
        )
    else:
        print('Use the normal SVD ...')

    # perform decomposition
    if args.first_eigen:
        print("\n --- use the first r eigen vecs as adapters --- \n")
    else:
        print("\n --- use the last r eigen vecs as adapters --- \n")
    build_model2(model, args)

    # save as hugging face model
    if args.save_model:
        assert args.save_path is not None
        save_path = args.save_path

        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        config = model.config.to_dict()
        config["lora_r"] = args.r
        config["auto_map"] = {
            "AutoConfig": "configuration_oursvd_qwen2.CovSVDQwen2Config",
            "AutoModelForCausalLM": "modeling_oursvd_qwen2.CovSVDQwen2ForCausalLM",
        }
        config["architectures"] = ["CovSVDQwen2ForCausalLM"]
        # 复制 Qwen2 版本的 mapping 文件到 save_path
        mapping_dir = os.path.join(os.path.dirname(__file__), "mapping")
        os.system(
            f"cp {mapping_dir}/configuration_oursvd_qwen2.py {mapping_dir}/modeling_oursvd_qwen2.py {save_path}/"
        )
        import json

        json.dump(config, open(save_path + "/config.json", "w"), indent=2)

        print(f"Done building huggingface model in {save_path}")
        del model
        del tokenizer
    # finished


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Pretrained model ID (from ModelScope or HuggingFace)",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd (ASVD)",
    )
    parser.add_argument(
        "--cov_aware",
        action="store_true",
    )
    parser.add_argument("--singular_aware",
                        action="store_true")
    parser.add_argument("--singular_aware_2",
                        action="store_true")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hyper-parameter alpha for ASVD",
    )
    parser.add_argument(
        "--calib_loader_size",
        type=int,
        default=256,
        help="number of samples used for covariance matrices",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb", "traivia_qa", "nqopen", "MetaMATH", "codefeedback", "WizLMinstruct", "alpaca"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max", "fisher", "fisher_abs_mean"],
        help="scaling method",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="use cached calibration results",
    )
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="evaluate mmlu",
    )
    parser.add_argument(
        "--sigma_fuse",
        type=str,
        default="UV",
        help="sigma fuse method",
        choices=["U", "V", "UV"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=233,
        help="random seed",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--first_eigen",
        action="store_true",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="build_adapters",
        choices=["full_decompose", "build_adapters"],
    )
    args = parser.parse_args()

    main(args)
