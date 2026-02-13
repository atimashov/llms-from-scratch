from datetime import datetime
from argparse import ArgumentParser, BooleanOptionalAction
import torch
from termcolor import colored, cprint
import yaml
from pathlib import Path
import subprocess

from p2_efficiency.profiling import nsys_profile_llm

def parse_config(cfg):
    meta = cfg.get("meta", {})
    run = cfg.get("run", {})
    sweep = cfg.get("sweep", {})

    seed = meta.get("seed", 123)
    mode = run.get("mode", "fwd_bcwd")
    script = meta.get("script_module", "p2_efficiency.scripts.run_nsys_profiler")
    trace = meta.get("trace", "cuda,nvtx,osrt")

    # AMP
    amp_cfg = run.get("amp", {})
    amp_enabled = amp_cfg.get("enabled", False)
    autocast_dtype = amp_cfg.get("autocast_dtype", "bfloat16")
    amp_suffix = "_amp" if amp_enabled else ""
    cmd_amp = f"--is-amp --autocast-dtype {autocast_dtype}" if amp_enabled else "--no-is-amp"

    # Cmd prefix
    cmd_prefix = f"uv run nsys profile --force-overwrite true -t {trace}" # not profiling CPU

    # Script prefix
    script_prefix = f"python -m {script} --seed {seed} --mode {mode} {cmd_amp}"

    # Output root
    out_root = Path(f"p2_efficiency/outputs/nsys_profiler/{mode}")

    # Sweep
    cmds = []
    for bs in sweep["batch_size"]:
        cmd_bs = f"--batch-size {bs}"
        for cntx_len in sweep["context_lengths"]:
            cmd_cntx = f"--context-length {cntx_len}"
            for nl, d, d_ff, h in sweep["models"]:
                base_dir = out_root / f"b_{bs}_s_{cntx_len}_l_{nl}_d_{d}_dff_{d_ff}_h_{h}"
                cmd_model = f"--num-layers {nl} --d-model {d} --d-ff {d_ff} --num-heads {h}"
                for attn_type in sweep["attn_types"]:
                    cmd_attn = f"--attn-type {attn_type}"
                    name = base_dir / f"{attn_type}{amp_suffix}"
                    if attn_type != "flash":
                        cmd_out = f'-o "{name}"'
                        cmd_script = f"{script_prefix} {cmd_bs} {cmd_cntx} {cmd_model} {cmd_attn}"
                        cmds.append(f"{cmd_prefix} {cmd_out} {cmd_script}")
                    else:
                        for q_tile, k_tile, n_warps, n_stages in sweep["gpu_params"]:
                            gpu_tag = f"qt_{q_tile}_kt_{k_tile}_nw_{n_warps}_ns_{n_stages}"
                            cmd_out = f'-o "{name}_{gpu_tag}"'
                            cmd_triton = f"--q-tile {q_tile} --k-tile {k_tile} --num-warps {n_warps} --num-stages {n_stages}"
                            cmd_script = f"{script_prefix} {cmd_bs} {cmd_cntx} {cmd_model} {cmd_attn} {cmd_triton}"
                            cmds.append(f"{cmd_prefix} {cmd_out} {cmd_script}")
    
    return cmds

def run_commands(cmds, dry_run: bool = True):
    fails = 0
    for i, cmd in enumerate(cmds, 1):
        start = cmd.find("--batch-size")
        cprint(f"[{i}/{len(cmds)}] {cmd[start:]}", "blue")
        if not dry_run:
            r = subprocess.run(cmd, shell = True)
            if r.returncode != 0:
                fails += 1
                cprint(f"[FAIL] exit_code = {r.returncode}", "red")
        print()
    print(colored(f"[DONE] total={len(cmds)};", "green"), colored(f"fails={fails}", "red" if fails > 0 else "green"))


if __name__ == '__main__':
    curr_time = datetime.now().time()
    cprint(f"⏱️ Run started at {curr_time.strftime('%H:%M:%S')}.", "red", attrs=["bold"])
    cprint("-" * 200, 'red', attrs=["bold"])

    # 1. Read config
    parser = ArgumentParser()
    parser.add_argument('--cfg-name', type=str, default='nsys_sweep')
    parser.add_argument('--dry-run', action=BooleanOptionalAction, default=True, help = "to choose 'False', use '--no-dry-run' flag")
    args = parser.parse_args()

    cfg_path = Path(f"p2_efficiency/configs/profiling/{args.cfg_name}.yaml")
    with open(cfg_path, 'r') as stream:
        cfg = yaml.safe_load(stream)

    # 2. Create list of commands
    cmds = parse_config(cfg)

    # 3. Run these commands
    run_commands(cmds, args.dry_run)