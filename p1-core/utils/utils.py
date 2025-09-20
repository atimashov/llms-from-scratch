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
    # force stability
    if torch.is_autocast_enabled():
        x = x.float()
    x_max = x.max(dim=dim, keepdim=True).values
    exps = torch.exp((x - x_max) / tau)
    return exps / torch.sum(exps, dim = dim, keepdim=True)


def mem(label=""):
    label = " " * (40 - len(label)) + label
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserv = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    max_reserv = torch.cuda.max_memory_reserved() / 1024**2
    print(f"{label} | allocated={alloc:.2f} MB, reserved={reserv:.2f} MB, "
          f"max_alloc={max_alloc:.2f} MB, max_reserved={max_reserv:.2f} MB")

def cross_entropy(logits: torch.Tensor, target: torch.Tensor, z_alpha:float = 0.0) -> torch.Tensor:
    assert logits.shape[:-1] == target.shape, "logits and target shape dimension mismatch"
    # Force stability: keep logits in fp32 under AMP
    if torch.is_autocast_enabled():
        logits = logits.float()
    # Flatten all dimensions except vocab
    logits_flat = rearrange(logits, "... v_size -> (...) v_size")
    target_flat = rearrange(target, "... -> (...)")

    # Subtract max logit per row
    m = logits_flat.max(dim=-1, keepdim=True).values
    logits_adj = logits_flat - m

    # Save stabilized logits
    B = logits_adj.shape[0]
    logits_idx = logits_adj[torch.arange(B, device=target.device), target_flat]
    log_z = torch.log(torch.sum(torch.exp(logits_adj), dim = -1))

    # Calculate loss: Negative log-likelihood and log_z
    losses = -logits_idx + log_z
    if z_alpha > 0:
        losses += z_alpha * log_z**2
    return torch.mean(losses)

def perplexity(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.exp(cross_entropy(logits, target, 0.0))

def cosine_lr_schedule(t: int, lr_max: float, lr_min: float, warmup_iters: int, flat_iters: int, cosine_cycle_iters: int):
    assert warmup_iters >= 0, f"Invalid warmup iterations: {warmup_iters}"
    assert  cosine_cycle_iters > warmup_iters, f"Invalid cosine cycle iterations: {cosine_cycle_iters}"
    # warm up
    if t < warmup_iters:
        return t / warmup_iters * lr_max
    # flat
    if t < flat_iters:
        return lr_max
    # cosine annealing
    if t < cosine_cycle_iters:
        return lr_min + 0.5 * (1 + math.cos((t - flat_iters) / (cosine_cycle_iters - flat_iters) * math.pi)) * (lr_max - lr_min)
    # post annealing
    return lr_min

def cosine_with_drops_lr_shedule(t: int, lr_max: float, lr_min: float, warmup_iters: int, flat_iters: int, cosine_cycle_iters: int, n_drops: int = 2, ratio: float = 0.1):
    # base cosine schedule
    base_lr = cosine_lr_schedule(t, lr_max, lr_min, warmup_iters, flat_iters, cosine_cycle_iters)

    # only apply drops after flat_iters
    if t < flat_iters:
        return base_lr

    # figure out which drop stage we’re in
    cosine_steps = cosine_cycle_iters - flat_iters
    stage = min(n_drops, (t - flat_iters) * (n_drops + 1) // cosine_steps)

    # scale learning rates
    return base_lr * (ratio ** stage)

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

def log_evals(ckpt_path: str, step: int, t_train, tokens, model, optimizer, loss_fn, config, use_amp):
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
    
    # return summary
    s_loss = f"Train loss = {final_t_loss:.4f} | Valid loss = {final_v_loss:.4f}"
    s_perp = f"Train perplexity = {final_t_perp:.4f} | Valid perplexity = {final_v_perp:.4f}"
    s_time = f"Train time = {perf_counter() - t_train:.2f}s | Eval time = {perf_counter() - t_eval:.2f}s"
    return f"{s_loss} | {s_perp} | Num of samples = {valid_total:,} | {s_time}"
    


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
        optimizer_params["betas"] = (config_opt["beta1"], config_opt["beta2"])
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
        "init_type": config["model"]["inits"]["type_ff"],
        "std_emb": config["model"]["inits"]["std_emb"],
        "clip_w": config["model"]["inits"]["clip_w"],
        "vocab_size": config["model"]["vocab_size"],
        "norms": config["model"]["norms"],
        "weights_tying": config["model"]["weights_tying"],
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
        flat_iters = warmup_iters + int(config["optimizer"]["scheduler"]["flat_iters"] * steps)
        cosine_cycle_iters= int(config["optimizer"]["scheduler"]["cosine_cycle_iters"] * steps)

        # optimizer parameters
        optimizer_params  = parse_optim(config["optimizer"])

        # scheduler parameters
        scheduler_params = {
            "t": 1,
            "lr_max": lr_max,
            "lr_min": lr_min,
            "warmup_iters": warmup_iters,
            "flat_iters": flat_iters,
            "cosine_cycle_iters":cosine_cycle_iters,
        }
        if config["optimizer"]["scheduler"]["name"] == "cosine_with_drops":
            scheduler_params["n_drops"] = config["optimizer"]["scheduler"]["n_drops"]
            scheduler_params["ratio"] = config["optimizer"]["scheduler"]["ratio"]


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
        model_str = f"cntx_{cntx}_numlayers_{num_layers}_dmodel_{d_model}_dff_{d_ff}_numheads_{num_heads}{rope_str}_{activation_str}_{dtype_str}{compile_str}{weights_tying_str}{postfix}"
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
        
        logger_name = config["logger"]
        assert logger_name in {"wandb", "tensorboard"}, f"Logger can be: 'wandb', and 'tensorboard'; but provided {logger_name}"

        serialize_freq = min(config["serialize"]["frequency"] // config["validate"]["frequency"], 1) * config["validate"]["frequency"]
        if config["model"]["dtype_amp"] == "bfloat16" and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16
            if config["model"]["dtype_amp"] == "bfloat16":
                config["model"]["dtype_amp"] == "bf16_notsup"


        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

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
            "z_alpha": config["train"]["z_alpha"],
            "logger_name": logger_name,
            "autocast_dtype": autocast_dtype,
            "device_name": device_name
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

def get_model_memory(config):
    """
    Calculate memory for model parameters (no bias, activations handled separately).
    If AMP is used, weight keep the same dtype, only activations change.

    Memory estimate:
        - float16/bf16: ~ 2 x [2 x V x d + 12 x L x d^2]
        - FP32              : ~ 4 x [2 x V x d + 12 x L x d^2]
    Returns:
        total_param_bytes, rope_buffer_bytes
    """
    # extract variables
    BYTES_IN_MB = 1024 ** 2
    d_model = config["model"]["d_model"]
    d_ff = config["model"]["d_ff"]
    seq_len = config["model"]["context_length"]
    ff_ratio = 3 if config["model"] else 2
    n_layers = config["model"]["num_layers"]
    n_heads = config["model"]["num_heads"]
    vocab_size = config["model"]["vocab_size"]
    norms = config["model"]["norms"]
    num_emb_matrices = 1 if config["model"]["weights_tying"] else 2
    dtype_bytes = 4 if config["model"]["dtype"] in {"float", "float32", "amp"} else 2
    
    # Embedding parameters and Final linear layer
    numel_emb = vocab_size * d_model * num_emb_matrices # NOTE: ~ V x d OR 2 x V x d

    # Norm size estimates
    rmsnorm, lnorm = d_model, 2 * d_model

    # Norms inside each transformer block
    numel_norms_tb = 0
    if norms["before"] in {"RMSNorm", "LayerNorm"}:
        numel_norms_tb += (rmsnorm if norms["before"] == "RMSNorm" else lnorm)  # (attn & ffn)
    if norms["after"] in {"RMSNorm", "LayerNorm"}:
        numel_norms_tb += (rmsnorm if norms["after"] == "RMSNorm" else lnorm)  # (attn & ffn)

    # Final norm
    numel_norms_f = 0
    if norms["final"] in {"RMSNorm", "LayerNorm"}:
        numel_norms_f += (rmsnorm if norms["final"] == "RMSNorm" else lnorm)

    # Transformer block parameters (per layer)
    numel_tb = 4 * d_model ** 2 + numel_norms_tb # P, Q, V, O
    numel_tb += ff_ratio * d_ff * d_model + numel_norms_tb # ~ 4d^2 + numel_norms_tb

    # Final Memory
    numel = numel_emb + numel_norms_f + n_layers * numel_tb
    memory = dtype_bytes * numel

    # RoPE buffer
    buffer = dtype_bytes * seq_len * d_model / n_heads 

    # Mask buffer
    buffer += seq_len ** 2
    
    return memory / BYTES_IN_MB, buffer / BYTES_IN_MB

def get_activations_memory(config):
    """
    Estimate expected memory usage (in MB) for activations.
    Input counted once (act_emb). Per layer: attention + FFN activations.

    Memory estimate:
        - AMP (float16/bf16): ~ 2 B x S x [3d + L x (18d + 2H x S) + 2V - 2HS]
        - FP32              : ~ 4 B x S x [2d + L x (16d + 2H x S) + V - HS]
    Returns:
        total_activation_MB
    """
    BYTES_IN_MB = 1024 ** 2
    # extract variables
    B = config["train"]["batch_size"]
    d_model = config["model"]["d_model"]
    d_ff = config["model"]["d_ff"]
    S = config["model"]["context_length"]
    V = config["model"]["vocab_size"]
    L = config["model"]["num_layers"]
    H = config["model"]["num_heads"]
    gated = config["model"]["is_gate"]
    norms = config["model"]["norms"]

    # memory per item
    dtype = config["model"]["dtype"]
    is_fp32 = dtype in {"float", "float32"}
    is_fp16 = dtype == "float16"
    is_amp  = dtype == "amp"
    bytes_model = 4 if (is_fp32) else 2
    bytes_up = 4 if (is_fp32 or is_amp) else 2

    # sizes reuse
    BSd = B * S * d_model
    BSf = B * S * d_ff
    BHS2 = B * H * S * S
    BSV = B * S * V
    
    # norms present
    n_before = int(norms["before"] in {"RMSNorm", "LayerNorm"})
    n_after = int(norms["after"] in {"RMSNorm", "LayerNorm"})
    n_final = int(norms["final"] in {"RMSNorm", "LayerNorm"})

    # 0. Embeddings activations (saved once)
    mem_emb = bytes_model * BSd
    
    # 1. Attention activations (per layer)
    act_qkv = 3 * BSd # no RoPE 
    act_attn = BSd
    act_proj = BSd
    act_wght = BHS2
    act_sm = 0 if is_amp else BHS2 # for some reason for AMP it is not counted

    mem_mha_model = bytes_model * L * (act_qkv + act_attn + act_proj)
    mem_mha_up = bytes_up * (L * (act_wght + act_sm) - act_wght)
    mem_mha_norm = bytes_up * L * (n_before + n_after) * BSd
    mem_mha = mem_mha_model + mem_mha_up + mem_mha_norm

    # 2. FF block activations (per layer)
    hidden_mult = 3 if gated else 2

    mem_ff_model = bytes_model * L * (hidden_mult * BSf + BSd)
    mem_ff_norm = bytes_up * L * (n_before + n_after) * BSd
    mem_ff = mem_ff_model + mem_ff_norm

    # 3. Final layer activations (looks like only logits saved in max dtype)
    mem_final = bytes_up * (BSV + BSd * n_final)

    # 4 Memory total
    mem = mem_emb + mem_mha + mem_ff + mem_final

    return mem / BYTES_IN_MB


def get_expected_memory(config):
    """
    Estimate expected memory usage (in MB) for training.
    
    Breakdown:
        I.   Model parameters
        II.  Gradients
        III. Optimizer state
        IV.  Input tokens
        V.   Activations (for backward)
    """
    # extract variables
    BYTES_IN_MB = 1024 ** 2
    bs = config["train"]["batch_size"]
    seq_len = config["model"]["context_length"]
    vocab_size = config["model"]["vocab_size"]
    dtype_bytes_input = 2 if vocab_size <= 65536 else 4
    
    # --------- I. model ---------
    memory_model, buffer_model = get_model_memory(config)
    
    # ------- II. gradients ------
    memory_grad = memory_model
    
    # --- III. optimizer state ---
    # supported 'Lion', 'Adam', 'AdamW', 'Adan'
    memory_os = memory_grad if config["optimizer"]["name"] in {'Lion'} else 2 * memory_grad 
    
    # --------- IV. Input --------
    memory_input = dtype_bytes_input * bs * seq_len / BYTES_IN_MB

    # ------ V. Activations ------
    memory_act = get_activations_memory(config)

    memory_steady = memory_model + buffer_model + memory_grad + memory_os + memory_input
    memory_peak = memory_steady + memory_act
    return {
        "steady_MB": memory_steady,
        "peak_MB": memory_peak,
        "model_MB": memory_model,
        "buffer_MB": buffer_model,
        "grad_MB": memory_grad,
        "optim_MB": memory_os,
        "input_MB": memory_input,
        "activations_MB": memory_act,
    }

def print_memory_stats(memories):
    print(
        f"{colored("Memory consumption, MB: ", 'blue')}"
        f"{colored('Peak=', 'blue')}{memories['peak_MB']:,.2f} | "
        f"{colored('Steady=', 'blue')}{memories['steady_MB']:,.2f} | "
        f"{colored('Inputs=', 'blue')}{memories['input_MB']:,.2f} | "        
        f"{colored('Model=', 'blue')}{memories['model_MB']:,.2f} | "
        f"{colored('Buffer=', 'blue')}{memories['buffer_MB']:,.2f} | "
        f"{colored('Grads=', 'blue')}{memories['grad_MB']:,.2f} | "
        f"{colored('Opt states=', 'blue')}{memories['optim_MB']:,.2f} | "
        f"{colored('Activations=', 'blue')}{memories['activations_MB']:,.2f} | "
    )

def est_forward_flops(config):
    """
    FLOPs estimate (forward pass):
    - Attention per layer: ~8BSD^2 + 4BS^2D
    - FFN per layer: (4 if non-gated else 6) BSDD_ff
    - Final projection: 2 BSDV

    If d_ff ~ 4d (non-gated) or 8/3d (gated), TOTAL FLOPS:
    - ~ 2BSD x [L x (12D + 2S)  + V]
    - (per token) ~ 2D x [L x (12D + 2S)  + V]
    """
    # extract variables
    B = config["train"]["batch_size"]
    d_model = config["model"]["d_model"]
    d_ff = config["model"]["d_ff"]
    S = config["model"]["context_length"]
    V = config["model"]["vocab_size"]
    L = config["model"]["num_layers"]
    gated = config["model"]["is_gate"]

    # Transformer block FLOPS (per layer)
    flops_attn = 4 * (2 * B * S * d_model ** 2) + 2 * (2 * B * S ** 2  * d_model)
    hidden_mult = 3 if gated else 2
    flops_ff = hidden_mult * 2 * B * S * d_model * d_ff # in both cases ~16BSD^2

    # Final Projection
    flops_proj = 2 * B * S * d_model * V
    
    return (L * (flops_attn + flops_ff) + flops_proj) // (B * S)