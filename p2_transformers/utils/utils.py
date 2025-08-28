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

def softmax(x: torch.Tensor, dim: int, tau: float = 1.0) -> torch.Tensor:
    assert -x.dim() <= dim < x.dim(), "Dimension is wrong"
    assert tau > 0, "Temperature must be positive."
    x_max = x.max(dim=dim, keepdim=True).values
    exps = torch.exp((x - x_max) / tau)
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

def gradient_clipping(params: Iterable[nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
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

def get_start_seqs(start_from: int | None, batch_size: int | None, x_len: int | None, in_memory_ids: np.ndarray | None, mode : str):
    assert mode in {"sample", "in_memory_ids"}
    if mode == "sample":
        start_seqs = random.randint(0, x_len, size=batch_size)[:, None] # NOTE: consider shuffle if I want without replacement
    elif mode == "in_memory_ids":
        start_seqs = in_memory_ids[start_from:start_from + batch_size][:, None]
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
    # if start_from is not None:
    #     start_seqs = np.arange(start_from, start_from + batch_size)[:, None]
    # else:
    #     start_seqs = random.randint(0, x.shape[0] - context_length, size=batch_size)[:, None] # NOTE: consider shuffle if I want without replacement
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

def save_checkpoint(model, optimizer, iteration, out_path, config = None):
    model_cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    obj = {
        "model": model_cpu_state,
        "optimizer": optimizer.state_dict(),
        "iter_number": iteration
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
    return obj["iter_number"], obj.get("config", None)


def eval_batch(x: npt.NDArray, model, loss_fn, context_length: int, batch_size: int, device):
    model.eval()
    with torch.no_grad():
        start_seqs = get_start_seqs(None, batch_size, x.shape[0] - context_length, None, "sample")
        tokens_curr, tokens_next = data_loading(x, context_length, start_seqs, device)
        logits = model(tokens_curr)
        loss = loss_fn(logits, tokens_next).item()
    model.train()
    return loss


def eval(x: npt.NDArray, model, loss_fn, context_length: int, batch_size: int, num_samples: int | None, device):
    max_available = x.shape[0] - context_length
    num_samples = max_available if num_samples == -1 else min(num_samples, max_available)
    assert num_samples > 0, "Not enough tokens for evaluation"

    model.eval()
    total_loss, total_items = 0, 0
    
    # create indices
    im_ids = np.arange(num_samples, dtype = np.int32)
    random.shuffle(im_ids)
    
    with torch.no_grad(), tqdm(range(0, num_samples, batch_size), leave = True) as loop:    
        for start_from in loop:
            curr_batch_size = min(batch_size, num_samples - start_from)
            start_seqs = get_start_seqs(start_from, curr_batch_size, None, im_ids, "in_memory_ids")
            tokens_curr, tokens_next = data_loading(x, context_length, start_seqs, device)
            logits = model(tokens_curr)
            loss = loss_fn(logits, tokens_next)

            total_loss += loss.item() * curr_batch_size
            total_items += curr_batch_size
            avg_loss = total_loss / total_items
            loop.set_postfix(valid_loss=avg_loss, total_items = total_items)
    model.train()
    return avg_loss

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
        optimizer_params["eps"] = float(config_opt["epsilon"]),
        optimizer_params["decoupled"]: config_opt["name"] == "AdamW"
    elif config_opt["name"] == "Adan":
        optimizer_params["betas"] = (config_opt["beta1"], config_opt["beta2"], config_opt["beta3"])
        optimizer_params["eps"] = float(config_opt["epsilon"])
    elif config_opt["name"] == "Lion":
        optimizer_params["beta"] = config_opt["beta"]
        optimizer_params["is_trust_ratio"] = config_opt["is_trust_ratio"]
        optimizer_params["nesterov"] = config_opt["nesterov"]
    return optimizer_params

def clean(x, precision: int = 0):
    s = f"{round(float(x), 7):.{precision}e}".rstrip('0').rstrip('.')
    base, exp = s.split("e")
    # strip leading 0
    exp = exp.lstrip("+0") if not exp.startswith('-') else '-' + exp[1:].lstrip("0")
    return f"{base}e{exp}"

def parse_config(config, mode: str = "train"):
    assert mode in {"train", "generate", "eval"}, f"We can parse only in the following modes: 'train', 'generate', but provided '{mode}'"
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

    if mode == "train":
        # run's and scheduler's variables
        bs, cntx = config["train"]["batch_size"], config["model"]["context_length"]
        steps = int(config["train"]["total_tokens_processed"] / (bs * cntx))
        lr_max, lr_min = float(config["optimizer"]["lr"]), float(config["optimizer"]["scheduler"]["lr_min"])
        warmup_iters = int(config["optimizer"]["scheduler"]["warmup_iters"] * steps)
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
        model_str = f"dmodel_{model_params['d_model']}_dff_{model_params['d_ff']}_numlayers_{model_params['num_layers']}_numheads_{model_params['num_heads']}_cntx_{cntx}"
        sched_name = config["optimizer"]["scheduler"]["name"]
        sched_str = f"{sched_name}/steps_{steps}/warmup_{warmup_iters}"
        optim_suffix = "_tr" if config["optimizer"]["is_trust_ratio"] else ""
        optim_name = config["optimizer"]["name"] + optim_suffix
        w_decay = optimizer_params["weight_decay"]
        optim_str = f"{optim_name}/lrmax{clean(lr_max)}_lrmin{clean(lr_min)}_wdecay{clean(w_decay)}"
        dataset_name = Path(config["dataset_path"]["prefix"]).name
        ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{dataset_name}/{device_name}/exp_bs_{bs}/{sched_str}/{optim_str}/{model_str}/{ts_str}"


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
            "loader_mode": config["train"]["loader_mode"],
        }
        return model_params, optimizer_params, scheduler_params, clip_grad_params, tokens_params, run_params
    if mode == "generate":
        model_path = Path(config["model"]["load_prefix"]).expanduser() / config["model"]["load_name"]
        tokenizer_params = {
            "input_path": None,
            "vocab_size": None,
            "special_tokens": config["tokenizer"]["special_tokens"]
        }
        vocab_merges_path = Path(config["tokenizer"]["files_path"]).expanduser()
        return model_params, model_path, tokenizer_params, vocab_merges_path
    if mode == "eval":
        model_path = Path(config["model"]["load_prefix"]).expanduser() / config["model"]["load_name"]
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