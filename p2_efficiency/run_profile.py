from argparse import ArgumentParser
from utils import profile_llm
import torch

if __name__ == '__main__':
    # read config
    parser = ArgumentParser()
    # parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    # parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--d-model', type=int, default=768)
    parser.add_argument('--d-ff', type=int, default=3072)
    parser.add_argument('--num-heads', type=int, default=12)
    parser.add_argument('--context-length', type=int, default=128)
    parser.add_argument('--mode', type=str, default="forward+backward")
    parser.add_argument('--is-amp', type=bool, default=True)
    parser.add_argument('--autocast-dtype', type=str, default="bfloat16")
    parser.add_argument('--profile-iters', type=int, default=10)
    inputs = parser.parse_args()
    assert inputs.autocast_dtype in {"bfloat16", "float16"}, f"Wrong dtype is provided: {inputs.dtype}"


    d_model = inputs.d_model
    d_ff = inputs.d_ff
    num_layers = inputs.num_layers
    num_heads = inputs.num_heads
    context_length = inputs.context_length 
    mode = inputs.mode
    is_amp = inputs.is_amp
    autocast_dtype = torch.bfloat16 if inputs.autocast_dtype == "bfloat16" else torch.bfloat16
    profile_iters = inputs.profile_iters

    # with open(inputs.config, 'r') as stream:
    #     config = yaml.safe_load(stream)

    profile_llm(
        d_model = d_model, d_ff = d_ff, num_layers = num_layers, num_heads = num_heads, context_length = context_length, 
        is_amp = is_amp, autocast_dtype = autocast_dtype, mode = mode, profile_iters = profile_iters
    )