__all__ = [
    "generate_random_data",
    "benchmark_llm",
]

import torch
import numpy as np
from timeit import default_timer as timer
from p1_core.models import TransformerLM
from p1_core.utils import cross_entropy


def generate_random_data(batch_size, context_length, vocab_size, device):
    tokens_curr = torch.randint(0, vocab_size, (batch_size, context_length), dtype = torch.long, device=device)
    tokens_next = torch.randint(0, vocab_size, (batch_size, context_length), dtype = torch.long, device=device)
    return tokens_curr, tokens_next

def benchmark_llm(
    d_model, d_ff, num_layers, num_heads, 
    context_length = 256, warmup_steps = 10, benchmark_steps = 100,
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
    if mode != "forward":
        model.train()
    else:
        model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = lambda logits, target: cross_entropy(logits, target)
    
    # Generate dummy inputs
    tokens_curr, tokens_next  = generate_random_data(batch_size, context_length, vocab_size, device)
    
    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = model(tokens_curr)
        loss = loss_fn(logits, tokens_next)
        if mode != "forward":
            loss.backward()
            optimizer.step()

    # benchmark
    times = []

    for _ in range(benchmark_steps):
        optimizer.zero_grad()
        torch.cuda.synchronize()

        if mode.startswith("forward"):
            start_time = timer()
        
        logits = model(tokens_curr)
        loss = loss_fn(logits, tokens_next)

        if mode == "forward":
            torch.cuda.synchronize()
            times.append((timer() - start_time) * 1000)
            continue
        
        if mode == "backward":
            start_time = timer()

        loss.backward()
        torch.cuda.synchronize()
        times.append((timer() - start_time) * 1000)

        optimizer.step()
    return np.mean(times), np.std(times)