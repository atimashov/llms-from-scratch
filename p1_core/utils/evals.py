__all__ = [
    "eval_batch",
    "eval",
    "log_evals"
]

import numpy as np
import numpy.typing as npt
import torch
from torch.amp import autocast
from tqdm import tqdm
from termcolor import colored
from time import perf_counter
from datetime import datetime
from contextlib import nullcontext

from .checkpointing import load_checkpoint, save_checkpoint
from .data import data_loading, get_start_seqs
from .model_stats import est_forward_flops

def eval_batch(x: npt.NDArray, model, loss_fn, context_length: int, bs: int, device):
    model.eval()
    with torch.no_grad():
        start_seqs = get_start_seqs(None, bs, x.shape[0] - context_length, None, "sample")
        tokens_curr, tokens_next = data_loading(x, context_length, start_seqs, device)
        logits = model(tokens_curr)
        loss = loss_fn(logits, tokens_next).item()
    model.train()
    return loss

def eval(step: int, x: npt.NDArray, model, loss_fn, context_length: int, bs: int, im_ids: npt.NDArray, device, tqdm_logs: bool = True, train_data = False, use_amp: bool = False):
    num_samples = im_ids.shape[0]
    assert num_samples > 0, "Not enough tokens for evaluation"

    model.eval()
    total_loss, total_items = 0, 0
    
    loop_range = range(0, num_samples, bs)
    if tqdm_logs:
        curr_time = datetime.now().time()
        desc = colored(f" Validation ({'train' if train_data else 'valid'}) at step {step + 1:,} started at {curr_time.strftime('%H:%M:%S')}", 'blue')
        loop_context = tqdm(loop_range, leave=True, desc=desc)
    else: 
        loop_context = nullcontext(loop_range)

    amp_ctx = autocast(device_type = "cuda") if use_amp else nullcontext()
    with torch.no_grad(), amp_ctx, loop_context as loop:    
        for start_from in loop:
            curr_bs = min(bs, num_samples - start_from)
            start_seqs = get_start_seqs(start_from, curr_bs, None, im_ids, "in_memory_ids")
            tokens_curr, tokens_next = data_loading(x, context_length, start_seqs, device)
            logits = model(tokens_curr)
            loss = loss_fn(logits, tokens_next)

            total_loss += loss.item() * curr_bs
            total_items += curr_bs
            avg_loss = total_loss / total_items
            if tqdm_logs:
                label = colored("train_loss", "blue") if train_data else colored("valid_loss", "green")
                loop.set_postfix(**{label: f"{avg_loss:.3f}", "total_items": total_items})
    model.train()
    return avg_loss

def log_evals(ckpt_path: str, step: int, t_train, tokens, model, optimizer, scaler, scheduler_params, loss_fn, im_ids_report, config, use_amp):
    t_eval = perf_counter()

    # Unpack variables
    valid_total = config["validate"]["report"]["num_samples"] if config["validate"]["report"]["num_samples"] >= 0 else tokens["valid"].shape[0]
    context_length = config["model"]["context_length"]
    bs = config["train"]["batch_size"]
    if config["device"] == "cpu":
        device = torch.device(config["device"])
    elif isinstance(config["device"], int):
        device = torch.device(f"cuda:{config["device"]}")
    
    # Load model to evaluate
    if not ckpt_path.exists():
        save_checkpoint(model, optimizer, -1, -1, -1, ckpt_path, config)
    n_iter, loss_ckpt, _, _ = load_checkpoint(ckpt_path, model, None, None, device)

    # Calculate evals
    loss_report_train = eval(step, tokens["train"], model, loss_fn, context_length, bs, im_ids_report["train_report"], device, train_data = True, use_amp = use_amp)
    perp_report_train = np.exp(loss_report_train)
    loss_report_valid = eval(step, tokens["valid"], model, loss_fn, context_length, bs, im_ids_report["valid_report"], device, use_amp = use_amp)
    perp_report_valid = np.exp(loss_report_valid)

    # Save checkpoint with losses
    save_checkpoint(model, optimizer, scaler, scheduler_params, n_iter, loss_ckpt, loss_report_valid, ckpt_path, config)
    
    # Return summary
    os_bs = config["train"]["optim_step_batch_size"]
    flops_per_token = est_forward_flops(config)
    total_flops = (step + 1) * os_bs * context_length * flops_per_token
    s_loss = f"Train loss = {loss_report_train:.4f} | Valid loss = {loss_report_valid:.4f}"
    s_perp = f"Train perplexity = {perp_report_train:.4f} | Valid perplexity = {perp_report_valid:.4f}"    
    elapsed_t, elapsed_v = perf_counter() - t_train, perf_counter() - t_eval
    min_t, min_v = int(elapsed_t // 60), int(elapsed_v // 60)
    sec_t, sec_v = int(elapsed_t % 60), int(elapsed_v % 60)
    s_time = f"Train time = {min_t} min. {sec_t} sec. | Eval time = {min_v} min. {sec_v} sec."
    return f"Total FlOPS = {total_flops:.2e} | {s_loss} | {s_perp} | Num of samples = {valid_total:,} | {s_time}"