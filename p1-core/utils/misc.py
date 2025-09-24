__all__ = [
    "cosine_lr_schedule",
    "cosine_with_drops_lr_shedule",
    "gradient_clipping",
    "get_optim",
    "compute_grad_norm"
]


import torch
import numpy as np
import math
from optimizers import Adam, Adan, Lion

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

def get_optim(optim_name, optim_params):
    assert optim_name in {"AdamW", "Adam", "Adan", "Lion"}, f"Currently supported: Adam, AdamW, Adan, and Lion; but provided {optim_name}"
    if optim_name in {"AdamW", "Adam"}:
        return Adam(**optim_params)
    elif optim_name == "Adan":
        return Adan(**optim_params)
    elif optim_name == "Lion":
        return Lion(**optim_params)

def compute_grad_norm(model):
    norms = torch.stack([
        p.grad.norm(2) for p in model.parameters() if p.grad is not None
    ])
    return torch.norm(norms).item()