__all__ = [
    "generate_random_inputs",
    "benchmark_llm",
    "nsys_profile_llm",
    "pt_profile_llm",
    "profile_llm_memory",
    "generate_random_qkv",
    "make_functions",
    "profile_func"
]

import torch
import torch.cuda.nvtx as nvtx
from torch.amp import autocast, GradScaler
from torch.profiler import profile, ProfilerActivity, schedule

import numpy as np
import pandas as pd
from timeit import default_timer as timer
from datetime import datetime
from typing import Callable
from pathlib import Path
from termcolor import colored
import json

import p1_core.layers.attention
from p1_core.layers.attention import scaled_dot_product_attention as sdpa
from p1_core.utils import cross_entropy
from .nsys_transformer import AnnotatedTransformerLM as TransformerLM
from .nsys_attn import annotated_scaled_dot_product_attention, flash_attention
from p2_efficiency.kernels.flashattn2 import FlashAttention2


def generate_random_inputs(batch_size, context_length, vocab_size, device):
    tokens_curr = torch.randint(0, vocab_size, (batch_size, context_length), dtype = torch.long, device=device)
    tokens_next = torch.randint(0, vocab_size, (batch_size, context_length), dtype = torch.long, device=device)
    return tokens_curr, tokens_next

def generate_random_qkv(b: int, h_q: int, h_kv: int, s: int, d: int, device: str = "cuda:0", dtype = torch.bfloat16, requires_grad = False):
    """
    Generating random inout (Q, K, V) to the attention. 
    Considering that Tensor cores uses bfloat16.
    """
    assert d % h_q ==0, f"d_model should be divisible by h_q, but {d} % {h_q} == {d % h_q}"
    assert h_q % h_kv ==0, f"d_model should be divisible by h_q, but {h_q} % {h_kv} == {h_q % h_kv}"
    dtype = torch.bfloat16 if dtype =="bfloat16" else torch.float16 if dtype =="float16" else torch.float32

    Q = torch.randn(b, h_q, s, d // h_q, device = device, dtype = dtype, requires_grad=requires_grad)
    K = torch.randn(b, h_kv, s, d // h_kv, device = device, dtype = dtype, requires_grad=requires_grad)
    V = torch.randn(b, h_kv, s, d // h_kv, device = device, dtype = dtype, requires_grad=requires_grad)

    return Q, K, V


def make_functions(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, gpu_params_sweep : list, bcwd = False):
    # 1. helper function
    def fwd_bwd(attn_fn, bcwd):
        if bcwd:
            Q.grad = None
            K.grad = None
            V.grad = None
        out = attn_fn(Q = Q, K = K, V = V, is_causal = True)
        if bcwd:
            loss = out.sum() # scalar
            loss.backward()

    # 1. Naive attention
    fn_naive = lambda: fwd_bwd(sdpa, bcwd)

    # 2. Compiled attention
    compiled = torch.compile(sdpa)
    fn_compiled = lambda: fwd_bwd(compiled, bcwd)

    # 3. FlashAttention2
    fns_flash = []
    for q_tile, k_tile, num_warps, num_stages in gpu_params_sweep:
        name = f"flash_qtile_{q_tile}_ktile_{k_tile}_warps_{num_warps}_stages_{num_stages}"
        def attn_flash(Q, K, V, is_causal, q_tile=q_tile, k_tile=k_tile, num_warps=num_warps, num_stages=num_stages):
            return FlashAttention2.apply(Q, K, V, is_causal, q_tile, k_tile, num_warps, num_stages)
        
        # fns_flash.append((name, lambda: fwd_bwd(attn_flash, bcwd)))
        fns_flash.append((name, (lambda attn=attn_flash: fwd_bwd(attn, bcwd))))
    
    return ('naive', fn_naive), ('compiled', fn_compiled), *fns_flash

def event_to_dict(e):
    row = {"name": None}
    for attr in dir(e):
        if attr.startswith("_") or attr.startswith("cuda_") or attr.startswith("self_cuda_"):
            continue
        try:
            val = getattr(e, attr)
        except Exception:
            continue
        if callable(val):
            continue
        if isinstance(val, (int, float, str, bool, type(None))):
            row[attr if attr not in "key" else "name"] = val
    return row

def events_to_df(events):
    return pd.DataFrame(event_to_dict(e) for e in events)


@nvtx.range("profile llm")
def nsys_profile_llm(
    batch_size: int, vocab_size: int, num_layers, d_model, d_ff, num_heads, context_length = 256,
    attn_type = "naive", mode = "fwd_bcwd", is_amp = False, autocast_dtype = None, iters_profile = 100
):
    """
    This function is intended to create .nsys files to be analysed in UI, so
    - torch.cuda.synchronize is unnecessary 
    - warmup can be skipped manually
    """
    assert mode in {"fwd", "fwd_bcwd"}, f"Unsupported mode: {mode}"
    assert torch.cuda.is_available(), "GPU is not available"
    
    # Replace to annotated attention /flash attention
    if attn_type == "naive":
        p1_core.layers.attention.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    elif attn_type == "compiled":
        # NOTE: there are different modes of compile
        p1_core.layers.attention.scaled_dot_product_attention = torch.compile(annotated_scaled_dot_product_attention)
    elif attn_type == "flash":
        p1_core.layers.attention.scaled_dot_product_attention = flash_attention
    
    # Init default params
    device = torch.device("cuda:0")
    dtype = torch.float32

    attn_params = {
        "type": "mha",
        "num_heads": num_heads,
        "num_heads_kv": num_heads,
    }    
    # Init model, optimizer,  and loss function
    with nvtx.range("model init"):
        model = TransformerLM(
                d_model, d_ff, attn_params, "SiLU", False, num_layers, 
                context_length = context_length, vocab_size = vocab_size,
                norms = {"before": "RMSNorm"}, device = device, dtype = dtype
            )
        if mode == "fwd":
            model.eval()
        else:
            model.train()
            
    if mode != "fwd":
        with nvtx.range("optimizer init"):
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = lambda logits, target: cross_entropy(logits, target)
        scaler = GradScaler('cuda') if is_amp else None
    
    # Generate dummy inputs
    with nvtx.range("generate tokens"):
        tokens_curr, tokens_next  = generate_random_inputs(batch_size, context_length, vocab_size, device)

    # Profile
    for i in range(iters_profile):
        with nvtx.range(f"step {i}"):
            if mode != "fwd":
                with nvtx.range("zero grad"):
                    optimizer.zero_grad()
            
            with autocast('cuda', enabled = is_amp, dtype=autocast_dtype):
                with nvtx.range("forward pass"):
                    if mode == "fwd":
                        with torch.no_grad():
                            logits = model(tokens_curr)
                    else:
                        logits = model(tokens_curr)

                if mode == "fwd":
                    continue

                with nvtx.range("loss calculation"):
                    loss = loss_fn(logits, tokens_next)


            with nvtx.range("backward pass"):
                if is_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            with nvtx.range("optimizer step"):
                if is_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()


def pt_profile_func(path: Path, name: str, fn: Callable, iters_warmup: int = 20, iters_profile: int = 100):
    path.mkdir(parents = True, exist_ok = True)
    
    # 1. Warmup
    for _ in range(iters_warmup):
        fn()
    torch.cuda.synchronize()

    # 2. Profile
    sched = schedule(wait=0, warmup=5, active=iters_profile, repeat=1) # explore this more
    with profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes = True,
        with_stack = False, # a bit overhead, but useful
        profile_memory = True,
        schedule = sched,
    ) as prof:
        for _ in range(iters_profile+5):
            fn()
            prof.step()
        torch.cuda.synchronize()
        
    # 3. Save agregated stats.
    events = prof.key_averages()
    events_df = events_to_df(events)
    events_df.to_csv(path / f"{name}.csv", index = False)

    table = events.table(sort_by="device_time_total", row_limit=100)
    with open(path / f"{name}.txt", "w") as f:
        f.write(table)


def pt_profile_llm(
    batch_size: int, d_model, num_heads:int, num_heads_kv:int, seq_len: int, 
    gpu_params: str, mode = "fwd_bcwd", iters_profile = 100, device: str = "cuda:0", dtype: str = "bfloat16"
):
    # 1. Create Data and Paths
    with_backward = mode == "fwd_bcwd"
    Q, K, V = generate_random_qkv(batch_size, num_heads, num_heads_kv, seq_len, d_model, device = device, dtype = dtype, requires_grad = with_backward)
    folder_name = f"b_{batch_size}_s_{seq_len}_d_{d_model}_hq_{num_heads}_hkv_{num_heads_kv}"
    path = Path(f"p2_efficiency/outputs/pt_profiler/{dtype}/{mode}") / folder_name

    # 2. Create functions to profile
    tags_fns = make_functions(Q, K, V, gpu_params, bcwd = with_backward)
    
    # 3. Profile versions of attention (Naive, Compiled, Flash with diff GPU params)
    for tag, fn in tags_fns:
        try:
            pt_profile_func(fn = fn, path = path, name = tag, iters_profile = iters_profile)
            print(colored(f"[DONE] {folder_name} | {tag}", "green"))
        except Exception as e:
            print(colored(f"[FAIL] {folder_name} | {tag} | {type(e).__name__}: {e}", "red"))
    print()


def benchmark_llm(
    d_model, d_ff, num_layers, num_heads, 
    context_length = 256, iters_warmup = 10, iters_benchmark = 100,
    mode = "forward"
):
    assert mode in {"forward", "forward+backward"}, f"Unsupported mode: {mode}"
    assert torch.cuda.is_available(), "GPU is not available"

    # Init default params
    batch_size = 4
    vocab_size = 10_000
    device = torch.device("cuda:0")
    dtype = torch.float32

    # Init model, optimizer,  and loss function
    model = TransformerLM(
            d_model, d_ff, num_heads, "SiLU", False, num_layers, 
            context_length = context_length, vocab_size = vocab_size,
            norms = {"before": "RMSNorm"}, device = device, dtype = dtype
        )
    if mode == "forward":
        model.eval()
    else:
        model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = lambda logits, target: cross_entropy(logits, target)
    
    # Generate dummy inputs
    tokens_curr, tokens_next  = generate_random_inputs(batch_size, context_length, vocab_size, device)
    
    # Warmup
    for _ in range(iters_warmup):
        optimizer.zero_grad()
        logits = model(tokens_curr)
        loss = loss_fn(logits, tokens_next)
        if mode != "forward":
            loss.backward()
            optimizer.step()

    # benchmark
    times = []

    for _ in range(iters_benchmark):
        optimizer.zero_grad()
        torch.cuda.synchronize()

        start_time = timer()
        
        logits = model(tokens_curr)
        loss = loss_fn(logits, tokens_next)

        if mode == "forward":
            torch.cuda.synchronize()
            times.append((timer() - start_time) * 1000)
            continue
        
        loss.backward()
        torch.cuda.synchronize()
        times.append((timer() - start_time) * 1000)

        optimizer.step()
    return np.mean(times), np.std(times)

def profile_llm_memory(
    d_model, d_ff, num_layers, num_heads, 
    context_length = 256, iters_warmup = 10, iters_profile = 100,
    mode = "forward"
):
    assert mode in {"forward", "backward", "forward+backward"}, f"Unsupported mode: {mode}"
    assert torch.cuda.is_available(), "GPU is not available"

    # Init default params
    batch_size = 4
    vocab_size = 10_000
    device = torch.device("cuda:0")
    dtype = torch.float32

    # Init model, optimizer,  and loss function
    model = TransformerLM(
            d_model, d_ff, num_heads, "SiLU", False, num_layers, 
            context_length = context_length, vocab_size = vocab_size,
            norms = {"before": "RMSNorm"}, device = device, dtype = dtype
        )
    if mode == "forward":
        model.eval()
    else:
        model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = lambda logits, target: cross_entropy(logits, target)
    
    # Generate dummy inputs
    tokens_curr, tokens_next  = generate_random_inputs(batch_size, context_length, vocab_size, device)
    
    # Warmup
    for _ in range(iters_warmup):
        optimizer.zero_grad()
        logits = model(tokens_curr)
        loss = loss_fn(logits, tokens_next)
        if mode != "forward":
            loss.backward()
            optimizer.step()

    # Start recording memory history.
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # Profile
    times = []
    for _ in range(iters_profile):
        optimizer.zero_grad()
        logits = model(tokens_curr)
        loss = loss_fn(logits, tokens_next)
        if mode == "forward":
            loss.backward()
            optimizer.step()

    # Save a pickle file to be loaded by PyTorch's online tool.
    torch.cuda.memory._dump_snapshot(f"memory_snapshot_{d_model}_{d_ff}_{num_layers}_{num_heads}_{context_length}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pickle")

    # Stop recording history.
    torch.cuda.memory._record_memory_history(enabled=None)

# ----------------------------------
#     Attention: PyTorch Profiler
# ----------------------------------
