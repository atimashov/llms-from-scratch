_all__ = [
    "log_stats",
    "log_gpu_stats",
    "log_train_stats",
    "log_valid_stats",
    "log_grad_norm_stats",
    "log_perf_stats",
]

import wandb
import torch
import numpy as np

def log_stats(log_data: dict, step: int, logger: str, writer):
    if logger == "wandb":
        wandb.log(log_data, step=step)
    elif logger == "tensorboard":
        assert writer is not None, "TensorBoard logger requires a writer instance."
        for k, v in log_data.items():
            writer.add_scalar(k, v, step)

def log_gpu_stats(log_data: dict, config: dict, name: str):
    used_mem = torch.cuda.memory_allocated(device=config["device"]) / (1024 ** 3)
    reserved_mem = torch.cuda.memory_reserved(device=config["device"]) / (1024 ** 3)
    max_used = torch.cuda.max_memory_allocated(device=config["device"]) / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved(device=config["device"]) / (1024 ** 3)
    log_data.update({
        f"perf_systems/gpu_mem_{name}/allocated_GB": used_mem,
        f"perf_systems/gpu_mem_{name}/reserved_GB": reserved_mem,
        f"perf_systems/gpu_mem_{name}/allocated_max_GB": max_used,
        f"perf_systems/gpu_mem_{name}/reserved_max_GB": max_reserved,
    })

def log_train_stats(log_data:dict, lr: float, train_loss: float, logits_norm_acc: float, logits_max_acc: float, logits_std_acc: float):
    log_data.update({
        "perf/lr": lr,
        "perf/train_loss": train_loss,
        "perf/train_perplexity": float('inf') if train_loss > 20 else np.exp(train_loss),
        "debug/logits_norm": logits_norm_acc,
        "debug/logits_max": logits_max_acc,
        "debug/logits_std": logits_std_acc,
    })

def log_valid_stats(log_data: dict, valid_loss: float):
    log_data.update({
        "perf/valid_loss": valid_loss,
        "perf/valid_perplexity": float('inf') if valid_loss > 20 else np.exp(valid_loss),
    })    

def log_grad_norm_stats(log_data: dict, g_norm_pre: float, g_norm_post: float):
    log_data.update({"debug/grad_norm_pre_clip": g_norm_pre})
    if g_norm_post is not None:
        log_data.update({"debug/grad_norm_post_clip": g_norm_post})

def log_perf_stats(log_data: dict, tokens_per_sec: float, flops_per_sec: float):
    log_data.update({
        "perf_systems/tokens_per_sec": tokens_per_sec,
        "perf_systems/forward_flops_per_sec": flops_per_sec,
    })