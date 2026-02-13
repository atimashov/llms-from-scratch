from datetime import datetime
from argparse import ArgumentParser, BooleanOptionalAction
import torch
from termcolor import colored, cprint
import yaml

from p2_efficiency.profiling import nsys_profile_llm

def run_experiment(args):
    # control source of randonmess
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high') # NOTE: what is it?

    # assert arguments
    assert args.autocast_dtype in {"bfloat16", "float16"}, f"Wrong dtype is provided: {args.autocast_dtype}"
    assert args.attn_type in {"naive", "compiled", "flash"}, f"Wrong attention type is provided: {args.attn_type}"
    assert (args.attn_type == "flash" and args.is_amp) or args.attn_type != "flash", f"For Flash Atention it is expected to use AMP"
    assert args.mode in {"fwd", "fwd_bcwd"}, f"Wrong mode is provided: {args.mode}"
    
    # Run profiling
    nsys_profile_llm(
        batch_size = args.batch_size,
        vocab_size = args.vocab_size,
        num_layers = args.num_layers,
        d_model = args.d_model,
        d_ff = args.d_ff,
        num_heads = args.num_heads, 
        context_length = args.context_length,
        is_amp = args.is_amp,
        autocast_dtype = torch.bfloat16 if args.autocast_dtype == "bfloat16" else torch.float16,
        attn_type = args.attn_type,
        mode = args.mode,
        iters_profile = args.iters_profile
    )

if __name__ == '__main__':
    curr_time = datetime.now().time()
    cprint(f"⏱️ Run started at {curr_time.strftime('%H:%M:%S')}.", "red", attrs=["bold"])
    cprint("-" * 200, 'red', attrs=["bold"])

    # read config
    parser = ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--vocab-size', type=int, default=10_000)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--d-model', type=int, default=768)
    parser.add_argument('--d-ff', type=int, default=3072)
    parser.add_argument('--num-heads', type=int, default=12)
    parser.add_argument('--context-length', type=int, default=128)
    parser.add_argument('--is-amp', action=BooleanOptionalAction, default=True, help = "to choose 'False', use '--no-is-amp' flag")
    parser.add_argument('--autocast-dtype', type=str, default="bfloat16")
    parser.add_argument('--attn-type', type=str, default='naive', help="'naive', 'compiled' or 'flash'")
    parser.add_argument('--q-tile', type=int, default=16)
    parser.add_argument('--k-tile', type=int, default=16)
    parser.add_argument('--num-warps', type=int, default=4)
    parser.add_argument('--num-stages', type=int, default=1)    
    parser.add_argument('--mode', type=str, default='fwd_bcwd', help="'fwd' or 'fwd_bcwd'")
    parser.add_argument('--iters-profile', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    inputs = parser.parse_args()

    # run entry point
    run_experiment(inputs)