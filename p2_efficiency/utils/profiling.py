__all__ = [
    "generate_random_data",
    "benchmark_llm",
    "profile_llm",
    "profile_llm_memory"
]

import torch
import numpy as np
from timeit import default_timer as timer
from datetime import datetime
import p1_core.layers.attention
from p2_efficiency.annotated import AnnotatedTransformerLM as TransformerLM
from p2_efficiency.annotated import annotated_scaled_dot_product_attention
from p1_core.utils import cross_entropy
import torch.cuda.nvtx as nvtx
from torch.amp import autocast, GradScaler


def generate_random_data(batch_size, context_length, vocab_size, device):
    tokens_curr = torch.randint(0, vocab_size, (batch_size, context_length), dtype = torch.long, device=device)
    tokens_next = torch.randint(0, vocab_size, (batch_size, context_length), dtype = torch.long, device=device)
    return tokens_curr, tokens_next

def benchmark_llm(
    d_model, d_ff, num_layers, num_heads, 
    context_length = 256, warmup_iters = 10, benchmark_iters = 100,
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
    tokens_curr, tokens_next  = generate_random_data(batch_size, context_length, vocab_size, device)
    
    # Warmup
    for _ in range(warmup_iters):
        optimizer.zero_grad()
        logits = model(tokens_curr)
        loss = loss_fn(logits, tokens_next)
        if mode != "forward":
            loss.backward()
            optimizer.step()

    # benchmark
    times = []

    for _ in range(benchmark_iters):
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

@nvtx.range("profile llm")
def profile_llm(
    d_model, d_ff, num_layers, num_heads, context_length = 256, mode = "forward", 
    is_amp = False, autocast_dtype = None, profile_iters = 100
):
    assert mode in {"forward", "forward+backward"}, f"Unsupported mode: {mode}"
    assert torch.cuda.is_available(), "GPU is not available"
    
    # Replace to annotated attention
    p1_core.layers.attention.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    
    # Init default params
    batch_size = 4
    vocab_size = 10_000
    device = torch.device("cuda:0")
    dtype = torch.float32

    # Init model, optimizer,  and loss function
    with nvtx.range("model init"):
        model = TransformerLM(
                d_model, d_ff, num_heads, "SiLU", False, num_layers, 
                context_length = context_length, vocab_size = vocab_size,
                norms = {"before": "RMSNorm"}, device = device, dtype = dtype
            )
        if mode == "forward":
            model.eval()
        else:
            model.train()
            
    if mode != "forward":
        with nvtx.range("optimizer init"):
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = lambda logits, target: cross_entropy(logits, target)
        scaler = GradScaler('cuda') if is_amp else None
    
    # Generate dummy inputs
    with nvtx.range("generate tokens"):
        tokens_curr, tokens_next  = generate_random_data(batch_size, context_length, vocab_size, device)

    # Profile
    for i in range(profile_iters):
        with nvtx.range(f"step {i}"):
            if mode != "forward":
                with nvtx.range("zero grad"):
                    optimizer.zero_grad()
            
            with autocast('cuda', enabled = is_amp, dtype=autocast_dtype):
                with nvtx.range("forward pass"):
                    if mode == "forward":
                        with torch.no_grad():
                            logits = model(tokens_curr)
                    else:
                        logits = model(tokens_curr)

                if mode == "forward":
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


def profile_llm_memory(
    d_model, d_ff, num_layers, num_heads, 
    context_length = 256, warmup_iters = 10, profile_iters = 100,
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
    tokens_curr, tokens_next  = generate_random_data(batch_size, context_length, vocab_size, device)
    
    # Warmup
    for _ in range(warmup_iters):
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
    for _ in range(profile_iters):
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

