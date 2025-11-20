__all__ = [
    "get_short_gpu_name",
    "clean",
    "parse_optim",
    "parse_config",
]

import torch
from pathlib import Path
from datetime import datetime

from .model_stats import est_forward_flops

def get_short_gpu_name(gpu_id=0):
    name = torch.cuda.get_device_name(gpu_id)
    for repl in ["NVIDIA", "Generation", "GeForce"]:
        name = name.replace(repl, "")
    return name.strip().replace(" ", "")

def clean(x, precision: int = 2):
    s = f"{round(float(x), 7):.{precision}e}".rstrip('0').rstrip('.')
    base, exp = s.split("e")
    # strip tailing 0
    base = base.rstrip('0').rstrip('.')
    # strip leading 0
    exp = exp.lstrip("+0") if not exp.startswith('-') else '-' + exp[1:].lstrip("0")
    return f"{base}e{exp}"

def parse_optim(config_opt):
    assert config_opt["name"] in {"AdamW", "Adam", "Adan", "Lion"}, f"Currently supported: Adam, AdamW, Adan, and Lion; but provided {config_opt["name"]}"
    optim_params = {
        "lr": float(config_opt["lr"]),
        "weight_decay": float(config_opt["weight_decay"]),
    }
    if config_opt["name"] in {"AdamW", "Adam"}:
        optim_params["betas"] = (config_opt["beta1"], config_opt["beta2"])
        optim_params["eps"] = float(config_opt["epsilon"])
        optim_params["decoupled"]: config_opt["name"] == "AdamW"
    elif config_opt["name"] == "Adan":
        optim_params["betas"] = (config_opt["beta1"], config_opt["beta2"], config_opt["beta3"])
        optim_params["eps"] = float(config_opt["epsilon"])
    elif config_opt["name"] == "Lion":
        optim_params["betas"] = (config_opt["beta1"], config_opt["beta2"])
        optim_params["is_trust_ratio"] = config_opt["is_trust_ratio"]
        optim_params["nesterov"] = config_opt["nesterov"]
    return optim_params

def parse_config(config, mode: str = "train"):
    assert mode in {"train", "generate", "eval"}, f"We can parse only in the following modes: 'train', 'generate', but provided '{mode}'"
    # create dtype map
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "amp": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16, 
        "float64": torch.float64,
        "double": torch.float64
    }
    
    # device
    if config["device"] == "cpu":
        device = torch.device(config["device"])
        device_name = "cpu"
    elif isinstance(config["device"], int):
        device = torch.device(f"cuda:{config["device"]}")
        device_name = get_short_gpu_name(config["device"])
    # TODO: add Mac's 'mps' support here
    else:
        raise Exception(f"Unexpected device: {config["device"]}")

    # model parameters
    assert config["model"]["dtype"] in dtype_map, f"Type you provided is not supported: {config["model"]["dtype"]}"
    assert config["model"]["activation"] in {"ReLU", "LeakyReLU", "SqReLU", "SiLU", "GELU"}, f"Type you provided is not supported: {config["model"]["activation"]}"
    d_model, d_ff = config["model"]["d_model"], config["model"]["d_ff"]
    attn_params = config["model"]["attention"]
    assert "type" in attn_params and "num_heads" in attn_params, f"Not complete set of params for attention, expected at least 'type' and 'num_heads'"
    assert attn_params["type"] in {"mha", "mqa", "gqa", "mla"}, f"Wrong attention type, supported only 'mha', 'mqa', 'gqa', 'mla'"
    if attn_params["type"] == "mha":
        attn_params["num_heads_kv"] = attn_params["num_heads"]
    if attn_params["type"] == "mqa":
        attn_params["num_heads_kv"] = 1
    activation, is_gate = config["model"]["activation"], config["model"]["is_gate"]
    num_layers, cntx = config["model"]["num_layers"], config["model"]["context_length"]
    model_params = {
        "d_model": d_model,
        "d_ff": d_ff,
        "attn_params": attn_params,
        "activation": activation, 
        "is_gate": is_gate,
        "num_layers": num_layers,
        "theta": config["model"]["rope_theta"],
        "context_length": cntx,
        "init_type": config["model"].get("inits", {}).get("type_ff"),
        "std_emb": config["model"].get("inits", {}).get("std_emb"),
        "clip_w": config["model"].get("inits", {}).get("clip_w"),
        "vocab_size": config["model"]["vocab_size"],
        "norms": config["model"]["norms"],
        "weights_tying": config["model"].get("weights_tying", False),
        "device": device,
        "dtype": dtype_map[config["model"]["dtype"]]
    }

    ckpt_path = None if config.get("train", {}).get("ckpt_load_from", None) is None else Path(config["train"]["ckpt_load_from"]).expanduser()
    if mode == "train":
        # run's and scheduler's variables
        assert "optim_step_batch_size" not in config["train"] or config["train"]["optim_step_batch_size"] % config["train"]["batch_size"] == 0, "'optim step batch size' should be divisible by 'batch size'"
        bs = config["train"]["batch_size"]
        os_bs = config["train"].get("optim_step_batch_size", config["train"]["batch_size"])
        
        # start from total FLOPS or total tokens
        flops_per_token = est_forward_flops(config)
        if config["train"]["total_flops"] is not None:
            steps =  int(float(config["train"]["total_flops"]) / (flops_per_token * os_bs * cntx))
            config["train"]["total_tokens_processed"] = steps * os_bs * cntx
        else:
            steps = (config["train"]["total_tokens_processed"] + os_bs * cntx - 1) // (os_bs * cntx)
            config["train"]["total_flops"] = steps * os_bs * cntx * flops_per_token

        lr_max, lr_min = float(config["optimizer"]["lr"]), float(config["scheduler"]["lr_min"])
        warmup_iters = config["scheduler"]["warmup_iters"]
        flat_iters = warmup_iters + int(config["scheduler"]["flat_iters"] * steps)
        cosine_cycle_iters= int(config["scheduler"]["cosine_cycle_iters"] * steps)

        # optimizer parameters
        optim_params  = parse_optim(config["optimizer"])

        # scheduler parameters
        scheduler_params = {
            "t": 1,
            "lr_max": lr_max,
            "lr_min": lr_min,
            "warmup_iters": warmup_iters,
            "flat_iters": flat_iters,
            "cosine_cycle_iters":cosine_cycle_iters,
        }
        if config["scheduler"]["name"] == "cosine_with_drops":
            scheduler_params["n_drops"] = config["scheduler"]["n_drops"]
            scheduler_params["ratio"] = config["scheduler"]["ratio"]

        # clip_grad
        clip_grad_params = {"max_norm": config["optimizer"].get("clip_gradient", {}).get("max_norm", None)}

        # tokens parameters
        assert "tokenized" in config["dataset_path"], f"You need pretokenize text first and provide path."
        prefix_path = Path(config["dataset_path"]["prefix"]).expanduser() / config["dataset_path"]["tokenized"]
        tokens_params = {
            "train": str(prefix_path / "train.npy"),
            "valid": str(prefix_path / "valid.npy")
        }

        # run parameters
        rope_str = "" if model_params["theta"] is not None else "_no_rope"
        activation_str = f"{'gated_' if is_gate else ''}{activation.lower()}"
        dtype_str = config['model']['dtype']
        compile_str = "_cmpl" if config["train"]["compile"] else ""
        weights_tying_str = "_wt" if config["model"]["weights_tying"] else ""
        postfix = "" if config["serialize"]["postfix"] == "" else f"_{config["serialize"]["postfix"]}"
        attn_type, nheads, nheads_kv = attn_params["type"], attn_params["num_heads"], attn_params.get("num_heads_kv", attn_params["num_heads"]) 
        model_str = f"cntx_{cntx}_numlayers_{num_layers}_dmodel_{d_model}_dff_{d_ff}_{attn_type}_nh_{nheads}_nh_kv_{nheads_kv}{rope_str}_{activation_str}_{dtype_str}{compile_str}{weights_tying_str}"
        sched_name = config["scheduler"]["name"]
        sched_str = f"{sched_name}/steps_{steps}/warmup_{warmup_iters}"
        optim_suffix = "_tr" if config["optimizer"]["is_trust_ratio"] else ""
        optim_name = config["optimizer"]["name"] + optim_suffix
        w_decay = optim_params["weight_decay"]
        optim_str = f"{optim_name}/lrmax{clean(lr_max)}_lrmin{clean(lr_min)}_wdecay{clean(w_decay)}"
        dataset_name = Path(config["dataset_path"]["prefix"]).name
        ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        loss_eval = "init" if ckpt_path is None else '{}'
        abl_str = f"z_{clean(config["loss"]["z_alpha"])}"
        
        logger_name = config["logger"]
        assert logger_name in {"wandb", "tensorboard", None}, f"Logger can be: 'wandb', 'tensorboard', and 'None'; but provided {logger_name}"

        if logger_name == "wandb":
            run_name = f"{dataset_name}/wandb/{ts_str}{postfix}"
        else:
            run_name = f"{dataset_name}/{abl_str}/{device_name}/exp_bs_{bs}_step_bs_{os_bs}/loss_{loss_eval}/{sched_str}/{optim_str}/{model_str}{postfix}/"
        
        if config["model"]["mixed_precision_dtype"] == "bfloat16" and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16
            if config["model"]["mixed_precision_dtype"] == "bfloat16":
                config["model"]["mixed_precision_dtype"] == "bf16_notsup"

        run_params = {
            "steps": steps,
            "bs": bs,
            "os_bs": os_bs,
            "context_length": cntx,
            "valid_freq": config["validate"]["monitor"]["frequency"],    
            "serialize_path": config["serialize"]["path"],
            "run_name": run_name,
            "device": device,
            "loader_mode": config["train"]["loader_mode"],
            "resume_training": config["train"].get("resume_training", False),
            "logger_name": logger_name,
            "autocast_dtype": autocast_dtype,
            "device_name": device_name,
        }
        return model_params, ckpt_path, optim_params, scheduler_params, clip_grad_params, tokens_params, run_params
    if mode == "generate":
        tokenizer_params = {
            "input_path": None,
            "vocab_size": None,
            "special_tokens": config["tokenizer"]["special_tokens"]
        }
        vocab_merges_path = Path(config["tokenizer"]["files_path"]).expanduser()
        return model_params, ckpt_path, tokenizer_params, vocab_merges_path
    if mode == "eval":
        tokens_path = Path(config["data"]["path"]).expanduser()
        return model_params, ckpt_path, tokens_path