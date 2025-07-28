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

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert -x.dim() <= dim < x.dim()
    x_max = x.max(dim=dim, keepdim=True).values
    exps = torch.exp(x - x_max)
    return exps / torch.sum(exps, dim = dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None):
    """
    Q, K:  (batch_size, ..., seq_len, d_k)
    V:  (batch_size, ..., seq_len, d_v)
    """
    d_k = K.shape[-1]
    # seq_len is the same for both, but I distinguish the ordering
    scores = einsum(Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") 
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    weights = softmax(scores / (d_k ** 0.5), dim = -1)
    att = einsum(weights, V, "... seq_len seq_len2, ... seq_len2 d_v -> ... seq_len d_v")
    return att

def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # TODO: assert -x.dim() <= dim < x.dim()
    # rearrange
    logits_2d = rearrange(logits, "... v_size -> (...) v_size")
    target_1d = rearrange(target, "... -> (...)")
    # substract the largest value
    adj_logits = logits_2d - logits_2d.max(dim=-1, keepdim=True).values
    # calculate loss
    exps = torch.exp(adj_logits)
    B = adj_logits.shape[0]
    losses = -adj_logits[torch.arange(B), target_1d] + torch.log(torch.sum(exps, dim = -1))
    return torch.mean(losses)

def perplexity(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.exp(cross_entropy(logits, target))

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

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    """
    returns pre-clipping gradient value for logging
    """
    assert max_l2_norm > 0, f"Max L2 norm should be positive but it is {max_l2_norm}."
    # get global norm
    sum_sq = None
    for param in params:
        g = param.grad
        if g is not None:
            if sum_sq is None:
                sum_sq = torch.zeros(1, device = g.device)
            sum_sq += g.detach().float().pow(2).sum()
    # check if no gradients or current norm is too large
    if sum_sq is None:
        return None
    global_l2_norm = sum_sq.sqrt()
    if global_l2_norm <= max_l2_norm:
        return global_l2_norm.item()
    # update gradients
    scale = max_l2_norm / (global_l2_norm + eps)
    with torch.no_grad():
        for param in params:
            if param.grad is not None:
                param.grad.mul_(scale.to(param.grad.dtype))
    return global_l2_norm.item()

def data_loading(x: npt.NDArray, batch_size: int, start_from: int | None, context_length: int, device: torch.device | None = None) -> (torch.Tensor, torch.Tensor):
    # create masks to sample from numpy
    if start_from is not None:
        start_seqs = np.arange(start_from, start_from + batch_size)[:, None]
    else:
        start_seqs = random.randint(0, x.shape[0] - context_length, size=batch_size)[:, None] # NOTE: consider shuffle if I want without replacement
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

def save_checkpoint(model, optimizer, iteration, out_path):
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_number": iteration
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, out_path)

def load_checkpoint(src, model, optimizer):
    obj = torch.load(src)
    # load state dicts
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iter_number"]

def get_valid_loss(x: npt.NDArray, model, loss_fn, context_length: int, batch_size: int, max_length: int | None, device):
    model.eval()
    total_loss = 0
    total_items = 0

    with torch.no_grad():
        if max_length is None:
            tokens_curr, tokens_next = data_loading(x, batch_size, None, context_length, device)
            logits = model(tokens_curr)
            loss = loss_fn(logits, tokens_next)
            avg_loss = loss.item()
        else: # TODO: consider shuffle
            end_loop = min(x.shape[0] - context_length, max_length)
            for i in range(0, end_loop, batch_size):
                curr_batch_size = min(batch_size, end_loop - i)
                tokens_curr, tokens_next = data_loading(x, curr_batch_size, i, context_length, device)
                logits = model(tokens_curr)
                loss = loss_fn(logits, tokens_next)
                # find cumulative loss
                total_loss += loss.item() * curr_batch_size
                total_items += curr_batch_size
            avg_loss = total_loss / total_items
    model.train()
    return avg_loss

def get_short_gpu_name(gpu_id=0):
    name = torch.cuda.get_device_name(gpu_id)
    for repl in ["NVIDIA", "Generation", "GeForce"]:
        name = name.replace(repl, "")
    return name.strip().replace(" ", "")

def parse_config(config):
    # create dtype map
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
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

    # run's and scheduler's variables
    bs, cntx = config["train"]["batch_size"], config["model"]["context_length"]
    steps = int(config["train"]["total_tokens_processed"] / (bs * cntx))
    lr_max, lr_min = float(config["optimizer"]["lr"]), float(config["optimizer"]["scheduler"]["lr_min"])
    warmup_iters = int(config["optimizer"]["scheduler"]["warmup_iters"] * steps)
    cosine_cycle_iters= int(config["optimizer"]["scheduler"]["cosine_cycle_iters"] * steps)

    # model parameters
    assert config["model"]["dtype"] in dtype_map, f"Type you provided is not supported: {config["model"]["dtype"]}"
    model_params = {
        "d_model": config["model"]["d_model"],
        "d_ff": config["model"]["d_ff"],
        "num_heads": config["model"]["num_heads"],
        "num_layers": config["model"]["num_layers"],
        "theta": config["model"]["rope_theta"],
        "context_length": config["model"]["context_length"],
        "vocab_size": config["model"]["vocab_size"],
        "device": device,
        "dtype": dtype_map[config["model"]["dtype"]]
    }

    # optimizer parameters
    assert config["optimizer"]["name"] in {"AdamW", "Adam"}, f"Currently supported only Adam and AdamW, but provided {config["optimizer"]["name"]}"
    optimizer_params = {
        # "params": model.parameters(),
        "lr": lr_max,
        "betas": (config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
        "weight_decay": config["optimizer"]["weight_decay"],
        "eps": float(config["optimizer"]["epsilon"]),
        "decoupled": config["optimizer"]["name"] == "AdamW",
    }

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
    model_str = f"dmodel_{model_params['d_model']}_dff_{model_params['d_ff']}_numlayers_{model_params['num_layers']}_numheads_{model_params['num_heads']}_cntx_{cntx}"
    optim_str = f"cosine_lrmax{lr_max}_lrmin{lr_min}_steps_{steps}_warmup_{warmup_iters}"
    dataset_name = Path(config["dataset_path"]["prefix"]).name
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{dataset_name}/{model_str}/{optim_str}/{device_name}/exp_bs_{bs}_{ts_str}"
    serialize_freq = max(config["serialize"]["frequency_steps"] // config["validate"]["frequency_steps"], 1) * config["validate"]["frequency_steps"]
    run_params = {
        "steps": steps,
        "batch_size": bs,
        "context_length": cntx,
        "valid_freq": config["validate"]["frequency_steps"],
        "valid_total": config["validate"]["num_samples"],        
        "serialize_path": config["serialize"]["path"],
        "serialize_first": config["serialize"]["first_save"],
        "serialize_freq":serialize_freq,
        "run_name": run_name,
        "device": device,
    }
    return model_params, optimizer_params, scheduler_params, clip_grad_params, tokens_params, run_params

