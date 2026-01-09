from datetime import datetime
from argparse import ArgumentParser
import torch
from termcolor import colored
import yaml

from p2_efficiency.utils import profile_llm

def main(config, random_seed = 123):
    # control source of randonmess
    torch.manual_seed(random_seed)
    torch.set_float32_matmul_precision('high')

    # parse parameters
    d_model = config["d_model"]
    d_ff = config["d_ff"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    context_length = config["context_length"] 
    attn_type = config.get("attn_type", "default")
    mode = config["mode"]
    is_amp = config["is_amp"]
    autocast_dtype = torch.bfloat16 if config.get("autocast_dtype", "bfloat16") == "bfloat16" else torch.float16
    profile_iters = config["profile_iters"]

    assert config.get("autocast_dtype", "bfloat16") in {"bfloat16", "float16"}, f"Wrong dtype is provided: {config.get("autocast_dtype", "bfloat16")}"

    profile_llm(
        d_model = d_model, d_ff = d_ff, num_layers = num_layers, num_heads = num_heads, context_length = context_length, 
        is_amp = is_amp, autocast_dtype = autocast_dtype, attn_type = attn_type, mode = mode, profile_iters = profile_iters
    )


if __name__ == '__main__':
    curr_time = datetime.now().time()
    print(colored(f"⏱️ Run started at {curr_time.strftime("%H:%M:%S")}.", 'red', attrs=["bold"]))
    print(colored("-" * 200, 'red', attrs=["bold"]))

    # read config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='p2_efficiency/configs/profile_base.yaml', help='config file')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    inputs = parser.parse_args()

    with open(inputs.config, 'r') as stream:
        config = yaml.safe_load(stream)

    main(config, inputs.seed)