import torch
from torch import nn
from einops import rearrange, einsum
import numpy as np
import numpy.typing as npt
from numpy import random
import math
from typing import Iterable
from time import perf_counter
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from optimizers import Adam, Adan, Lion
from termcolor import colored
from contextlib import nullcontext
from torch.amp import autocast

def softmax(x: torch.Tensor, dim: int, tau: float = 1.0) -> torch.Tensor:
    assert -x.dim() <= dim < x.dim(), "Dimension is wrong"
    assert tau > 0, "Temperature must be positive."
    x_max = x.max(dim=dim, keepdim=True).values
    exps = torch.exp((x - x_max) / tau)
    return exps / torch.sum(exps, dim = dim, keepdim=True)

def cross_entropy(logits: torch.Tensor, target: torch.Tensor, z_alpha:float = 0.0) -> torch.Tensor:
    assert logits.shape[:-1] == target.shape, "logits and target shape dimension mismatch"
    # Flatten all dimensions except vocab
    logits_flat = rearrange(logits, "... v_size -> (...) v_size")
    target_flat = rearrange(target, "... -> (...)")
    
    # Numerical stability: subtract max logit per row
    logits_adj = logits_flat - logits_flat.max(dim=-1, keepdim=True).values
    
    # Log-sum-exp trick
    log_z = torch.log(torch.sum(torch.exp(logits_adj), dim = -1))
    B = logits_flat.shape[0]
    logits_idx = logits_adj[torch.arange(B), target_flat]

    # Calculate loss: Negative log-likelihood and log_z
    losses = -logits_idx + log_z
    if z_alpha > 0:
        losses += z_alpha * log_z**2
    return torch.mean(losses)

def perplexity(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.exp(cross_entropy(logits, target, 0.0))

def cosine_lr_schedule(t: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int):
    assert warmup_iters >= 0, f"Invalid warmup iterations: {warmup_iters}"
    assert  cosine_cycle_iters > warmup_iters, f"Invalid cosine cycle iterations: {cosine_cycle_iters}"
    # warm up
    if t < warmup_iters:
        lr = t / warmup_iters * lr_max
    # annealing
    elif t <= cosine_cycle_iters:
        lr = lr_min + 0.5 * (1 + math.cos((t - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (lr_max - lr_min)
    # post annealing
    else:
        lr = lr_min
    return lr

def gradient_clipping(model, max_l2_norm: float, eps: float = 1e-6):
    """
    returns pre-clipping and after-clipping gradient value for logging
    """
    assert max_l2_norm > 0, f"Max L2 norm should be positive but it is {max_l2_norm}."
    # get global norm
    sum_sq = None
    for param in model.parameters():
        g = param.grad
        if g is not None:
            if sum_sq is None:
                sum_sq = torch.zeros(1, device = g.device)
            sum_sq += g.detach().float().pow(2).sum()
    # check if no gradients or current norm is too large
    if sum_sq is None:
        return None, None
    global_l2_norm = sum_sq.sqrt()
    if global_l2_norm <= max_l2_norm:
        return global_l2_norm.item(), global_l2_norm.item()
    # update gradients
    scale = max_l2_norm / (global_l2_norm + eps)
    grad_device = global_l2_norm.device
    sum_sq = torch.zeros(1, device = grad_device)
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale.to(param.grad.dtype))
                sum_sq += param.grad.detach().float().pow(2).sum()
    return global_l2_norm.item(), sum_sq.sqrt().item()

def get_start_seqs(start_from: int | None, batch_size: int | None, x_len: int | None, in_memory_ids: np.ndarray | None, mode : str):
    assert mode in {"sample", "in_memory_ids"}
    if mode == "sample":
        start_seqs = random.randint(0, x_len, size=batch_size)[:, None]
    elif mode == "in_memory_ids":
        start_from = start_from % in_memory_ids.shape[0] 
        start_seqs = in_memory_ids[start_from:start_from + batch_size]
        if start_from + batch_size > in_memory_ids.shape[0]:
            offset = start_from + batch_size - in_memory_ids.shape[0]
            start_seqs = np.concat((start_seqs, in_memory_ids[:offset]), axis = 0)
    start_seqs = start_seqs[:, None]
    return start_seqs

def data_loading(x: npt.NDArray, context_length: int, start_seqs: np.ndarray, device: torch.device | None = None) -> (torch.Tensor, torch.Tensor):
    """
    Create batch of data to train.

    Args:

    Returns:
        tokens_curr:
        tokens_next:

    """
    # create masks to sample from numpy
    steps_curr = np.arange(context_length)[None, :]
    steps_next = np.arange(1, context_length + 1)[None, :]
    mask_curr, mask_next = start_seqs + steps_curr, start_seqs + steps_next
    # sample numpy tokens
    tokens_curr_np = x[mask_curr]
    tokens_next_np = x[mask_next]
    # convert to PyTorch (NOTE: how dtype conversion slows this down?)
    tokens_curr = torch.from_numpy(tokens_curr_np).to(device = device, dtype = torch.int)
    tokens_next = torch.from_numpy(tokens_next_np).to(device = device, dtype = torch.int)
    return tokens_curr, tokens_next

def save_checkpoint(model, optimizer, iteration, loss_step, loss_full, out_path, config = None):
    model_cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    obj = {
        "model": model_cpu_state,
        "optimizer": optimizer.state_dict(),
        "iter_number": iteration,
        "loss_step": loss_step,
        "loss_full": loss_full,
    }
    if config is not None:
        obj["config"] = config
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, out_path)

def load_checkpoint(src, model, optimizer, device = "cpu"):
    obj = torch.load(src, map_location = device)
    # load state dicts
    model.load_state_dict(obj["model"])
    if optimizer:
        optimizer.load_state_dict(obj["optimizer"])
    return obj["iter_number"], obj.get("loss_step", None), obj.get("loss_full", None), obj.get("config", None)


def eval_batch(x: npt.NDArray, model, loss_fn, context_length: int, batch_size: int, device):
    model.eval()
    with torch.no_grad():
        start_seqs = get_start_seqs(None, batch_size, x.shape[0] - context_length, None, "sample")
        tokens_curr, tokens_next = data_loading(x, context_length, start_seqs, device)
        logits = model(tokens_curr)
        loss = loss_fn(logits, tokens_next).item()
    model.train()
    return loss


def eval(step: int, x: npt.NDArray, model, loss_fn, context_length: int, batch_size: int, num_samples: int | None, device, tqdm_logs: bool = True, train_data = False, use_amp: bool = False):
    max_available = x.shape[0] - context_length
    num_samples = max_available if num_samples == -1 else min(num_samples, max_available)
    assert num_samples > 0, "Not enough tokens for evaluation"

    model.eval()
    total_loss, total_items = 0, 0
    
    # create indices (just first 'num_samples')
    im_ids = np.arange(num_samples, dtype = np.int32)
    
    loop_range = range(0, num_samples, batch_size)
    if tqdm_logs:
        curr_time = datetime.now().time()
        desc = colored(f" Validation ({'train' if train_data else 'valid'}) at step {step + 1:,} started at {curr_time.strftime('%H:%M:%S')}", 'blue')
        loop_context = tqdm(loop_range, leave=True, desc=desc)
    else: 
        loop_context = nullcontext(loop_range)

    amp_ctx = autocast(device_type = "cuda") if use_amp else nullcontext()
    with torch.no_grad(), amp_ctx, loop_context as loop:    
        for start_from in loop:
            curr_batch_size = min(batch_size, num_samples - start_from)
            start_seqs = get_start_seqs(start_from, curr_batch_size, None, im_ids, "in_memory_ids")
            tokens_curr, tokens_next = data_loading(x, context_length, start_seqs, device)
            logits = model(tokens_curr)
            loss = loss_fn(logits, tokens_next)

            total_loss += loss.item() * curr_batch_size
            total_items += curr_batch_size
            avg_loss = total_loss / total_items
            if tqdm_logs:
                label = colored("train_loss", "blue") if train_data else colored("valid_loss", "green")
                loop.set_postfix(**{label: f"{avg_loss:.3f}", "total_items": total_items})
    model.train()
    return avg_loss

def log_evals(ckpt_path: str, step: int, t_train, tokens, model, optimizer, loss_fn, config, writer, use_amp):
    t_eval = perf_counter()

    # unpack variables
    valid_total = config["validate"]["num_samples"] if config["validate"]["num_samples"] >= 0 else tokens["valid"].shape[0]
    context_length = config["model"]["context_length"]
    batch_size = config["train"]["batch_size"]
    if config["device"] == "cpu":
        device = torch.device(config["device"])
    elif isinstance(config["device"], int):
        device = torch.device(f"cuda:{config["device"]}")
    
    # load model to evaluate
    if not ckpt_path.exists():
        save_checkpoint(model, optimizer, -1, -1, -1, ckpt_path, config)
    n_iter, loss_step, _, _ = load_checkpoint(ckpt_path, model, None, device)

    # calculate evals
    final_t_loss = eval(step, tokens["train"], model, loss_fn, context_length, batch_size, valid_total, device, train_data = True, use_amp = use_amp)
    final_t_perp = np.exp(final_t_loss)
    final_v_loss = eval(step, tokens["valid"], model, loss_fn, context_length, batch_size, valid_total, device, use_amp = use_amp)
    final_v_perp = np.exp(final_v_loss)

    # save checkpoint with losses
    save_checkpoint(model, optimizer, n_iter, loss_step, final_v_loss, ckpt_path, config)
    
    # add results to my writer
    writer.add_text(
        "summary/final_valid_metrics",
        f"Train loss = {final_t_loss:.4f} | Valid loss = {final_v_loss:.4f} | "
        f"Train perplexity = {final_t_perp:.4f} | Valid perplexity = {final_v_perp:.4f} | "
        f"Num of samples = {valid_total:,} | "
        f"Train time = {perf_counter() - t_train:.2f}s | Eval time = {perf_counter() - t_eval:.2f}s",
        step+1
    )


def get_short_gpu_name(gpu_id=0):
    name = torch.cuda.get_device_name(gpu_id)
    for repl in ["NVIDIA", "Generation", "GeForce"]:
        name = name.replace(repl, "")
    return name.strip().replace(" ", "")

def parse_optim(config_opt):
    assert config_opt["name"] in {"AdamW", "Adam", "Adan", "Lion"}, f"Currently supported: Adam, AdamW, Adan, and Lion; but provided {config_opt["name"]}"
    optimizer_params = {
        "lr": float(config_opt["lr"]),
        "weight_decay": float(config_opt["weight_decay"]),
    }
    if config_opt["name"] in {"AdamW", "Adam"}:
        optimizer_params["betas"] = (config_opt["beta1"], config_opt["beta2"])
        optimizer_params["eps"] = float(config_opt["epsilon"])
        optimizer_params["decoupled"]: config_opt["name"] == "AdamW"
    elif config_opt["name"] == "Adan":
        optimizer_params["betas"] = (config_opt["beta1"], config_opt["beta2"], config_opt["beta3"])
        optimizer_params["eps"] = float(config_opt["epsilon"])
    elif config_opt["name"] == "Lion":
        optimizer_params["beta"] = config_opt["beta"]
        optimizer_params["is_trust_ratio"] = config_opt["is_trust_ratio"]
        optimizer_params["nesterov"] = config_opt["nesterov"]
    return optimizer_params

def clean(x, precision: int = 2):
    s = f"{round(float(x), 7):.{precision}e}".rstrip('0').rstrip('.')
    base, exp = s.split("e")
    # strip tailing 0
    base = base.rstrip('0').rstrip('.')
    # strip leading 0
    exp = exp.lstrip("+0") if not exp.startswith('-') else '-' + exp[1:].lstrip("0")
    return f"{base}e{exp}"

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
    d_model, d_ff, num_heads = config["model"]["d_model"], config["model"]["d_ff"], config["model"]["num_heads"]
    activation, is_gate = config["model"]["activation"], config["model"]["is_gate"]
    num_layers, cntx = config["model"]["num_layers"], config["model"]["context_length"]
    model_params = {
        "d_model": d_model,
        "d_ff": d_ff,
        "num_heads": num_heads,
        "activation": activation, 
        "is_gate": is_gate,
        "num_layers": num_layers,
        "theta": config["model"]["rope_theta"],
        "context_length": cntx,
        "vocab_size": config["model"]["vocab_size"],
        "norms": config["model"]["norms"],
        "device": device,
        "dtype": dtype_map[config["model"]["dtype"]]
    }
    model_path = None if "load_prefix" not in config["model"] else Path(config["model"]["load_prefix"]).expanduser() / config["model"]["load_name"]
    if mode == "train":
        # run's and scheduler's variables
        assert "optim_step_batch_size" not in config["train"] or config["train"]["optim_step_batch_size"] % config["train"]["batch_size"] == 0, "'optim step batch size' should be divisible by 'batch size'"
        bs = config["train"]["batch_size"]
        os_bs = config["train"].get("optim_step_batch_size", config["train"]["batch_size"])
        steps = (config["train"]["total_tokens_processed"] + os_bs * cntx - 1) // (os_bs * cntx)
        lr_max, lr_min = float(config["optimizer"]["lr"]), float(config["optimizer"]["scheduler"]["lr_min"])
        warmup_iters = config["optimizer"]["scheduler"]["warmup_iters"]
        cosine_cycle_iters= int(config["optimizer"]["scheduler"]["cosine_cycle_iters"] * steps)

        # optimizer parameters
        optimizer_params  = parse_optim(config["optimizer"])

        # scheduler parameters
        scheduler_params = {
            "t": 1,
            "lr_max": lr_max,
            "lr_min": lr_min,
            "warmup_iters": warmup_iters,
            "cosine_cycle_iters":cosine_cycle_iters,
        }

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
        model_str = f"cntx_{cntx}_numlayers_{num_layers}_dmodel_{d_model}_dff_{d_ff}_numheads_{num_heads}{rope_str}_{activation_str}_{dtype_str}{compile_str}"
        sched_name = config["optimizer"]["scheduler"]["name"]
        sched_str = f"{sched_name}/steps_{steps}/warmup_{warmup_iters}"
        optim_suffix = "_tr" if config["optimizer"]["is_trust_ratio"] else ""
        optim_name = config["optimizer"]["name"] + optim_suffix
        w_decay = optimizer_params["weight_decay"]
        optim_str = f"{optim_name}/lrmax{clean(lr_max)}_lrmin{clean(lr_min)}_wdecay{clean(w_decay)}"
        dataset_name = Path(config["dataset_path"]["prefix"]).name
        ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        loss_eval = "init" if model_path is None else '{}'
        abl_str = f"z_{clean(config["train"]["z_alpha"])}"
        run_name = f"{dataset_name}/{abl_str}/{device_name}/exp_bs_{bs}_step_bs_{os_bs}/loss_{loss_eval}/{sched_str}/{optim_str}/{model_str}/{ts_str}"


        serialize_freq = min(config["serialize"]["frequency"] // config["validate"]["frequency"], 1) * config["validate"]["frequency"]
        run_params = {
            "steps": steps,
            "batch_size": bs,
            "optimizer_step_batch_size": os_bs,
            "context_length": cntx,
            "valid_freq": config["validate"]["frequency"],
            "valid_total": config["validate"]["num_samples"],        
            "serialize_path": config["serialize"]["path"],
            # "serialize_first": config["serialize"]["first_save"],
            "serialize_freq":serialize_freq,
            "run_name": run_name,
            "device": device,
            "loader_mode": config["train"]["loader_mode"],
            "z_alpha": config["train"]["z_alpha"]
        }
        return model_params, model_path, optimizer_params, scheduler_params, clip_grad_params, tokens_params, run_params
    if mode == "generate":
        tokenizer_params = {
            "input_path": None,
            "vocab_size": None,
            "special_tokens": config["tokenizer"]["special_tokens"]
        }
        vocab_merges_path = Path(config["tokenizer"]["files_path"]).expanduser()
        return model_params, model_path, tokenizer_params, vocab_merges_path
    if mode == "eval":
        tokens_path = Path(config["data"]["path"]).expanduser()
        return model_params, model_path, tokens_path

def get_optim(optim_name, optim_params):
    assert optim_name in {"AdamW", "Adam", "Adan", "Lion"}, f"Currently supported: Adam, AdamW, Adan, and Lion; but provided {optim_name}"
    if optim_name in {"AdamW", "Adam"}:
        return Adam(**optim_params)
    elif optim_name == "Adan":
        return Adan(**optim_params)
    elif optim_name == "Lion":
        return Lion(**optim_params)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_grad_norm(model):
    norms = torch.stack([
        p.grad.norm(2) for p in model.parameters() if p.grad is not None
    ])
    return torch.norm(norms).item()

def print_d_model_d_ff(d_model: int, d_ff: int, is_gate: bool):
    ratio = 8/3 if is_gate else 4
    deviation = 100 * (max(d_ff / d_model, ratio) / min(d_ff / d_model, ratio) - 1)
    dev_print = colored(f"Deviaton is {deviation:.1f}%", "red" if deviation > 5 else  "green")
    draft_print = colored("Activation ", "blue") + f"is {'' if is_gate else 'not '}gated - d_ff/d_model is expected ~{ratio:.2f}; {dev_print}. "
    rec = 'good' if d_ff % 64 == 0 else 'not recommended'
    div64_print = f"Hidden dim is {'' if d_ff % 64 == 0 else 'not '}divisible by 64. It is {rec} for GPU."
    print(draft_print + div64_print)
    