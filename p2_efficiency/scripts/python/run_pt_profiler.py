from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
import torch
from termcolor import colored
import yaml

from p2_efficiency.profiling import pt_profile_llm


def parse_config(cfg):
    # 1. Meta configs
    device = cfg.get("meta", {}).get("device", "cuda:0")
    dtype = cfg.get("meta", {}).get("dtype", "bfloat16")
    seed = cfg.get("meta", {}).get("seed", 123)
    
    # 2. Run configs
    mode = cfg.get("meta", {}).get("mode", 'fwd_bcwd')
    iters_warmup = cfg.get("meta", {}).get("iters_warmup", 5)
    iters_profile = cfg.get("meta", {}).get("iters_profile", 5)
    # NOTE: It might make sense to add what variation ot profile

    # 3. Sweep configs
    model_params = cfg.get("sweep", {}).get("model_params", [(16, 256, 8, 8, 32)]) # batch_size, d_model, num_heads, num_heads_kv, seq_len
    gpu_params = cfg.get("sweep", {}).get("gpu_params", [(16, 16, 4, 1)]) # q_tile, k_tile, num_warps, num_stages
    is_causal = cfg.get("sweep", {}).get("is_causal", True) 

    return device, dtype, seed, mode, iters_warmup, iters_profile, model_params, model_params, gpu_params, is_causal


def run_experiment(
    seed: int, batch_size: int, context_length: int, d_model: int, h: int, h_kv: int,
    gpu_params: list, mode: str, iters_profile: int, device: torch.device, dtype: torch.dtype
    ):
    # 1. Control source of randonmess
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('high')

    # 2. Assert arguments
    assert mode in {"fwd", "fwd_bcwd"}, f"Wrong mode is provided: {args.mode}"
    
    # 3. Run profiling
    pt_profile_llm(
        batch_size = batch_size,
        d_model = d_model,
        num_heads = h, 
        num_heads_kv = h_kv, 
        seq_len = context_length,
        gpu_params = gpu_params,
        mode = mode,
        iters_profile = iters_profile,
        device = device,
        dtype = dtype,
    )

if __name__ == '__main__':
    curr_time = datetime.now().time()
    print(colored(f"⏱️ Run started at {curr_time.strftime('%H:%M:%S')}.", "red", attrs=["bold"]))
    print(colored("-" * 200, 'red', attrs=["bold"]))

    # 1. Read config
    parser = ArgumentParser()
    parser.add_argument('--cfg-name', type=str, default='pt_params_fwd_bcwd')
    inputs = parser.parse_args()

    cfg_path = Path(f"p2_efficiency/configs/profiling/{inputs.cfg_name}.yaml")
    with open(cfg_path, 'r') as stream:
        cfg = yaml.safe_load(stream)

    # 2. Parse configs
    device, dtype, seed, mode, iters_warmup, iters_profile, model_params, model_params, gpu_params, is_causal = parse_config(cfg)
    
    # 3. Run experiments
    for batch_size, cntx_len, d_model, h, h_kv in model_params:
        run_experiment(
            seed = seed,
            batch_size = batch_size,
            context_length = cntx_len,
            d_model = d_model,
            h = h,
            h_kv = h_kv,
            gpu_params = gpu_params,
            mode = mode,
            iters_profile = iters_profile,
            device = device,
            dtype = dtype,
        )