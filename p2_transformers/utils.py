import torch
from torch import nn
from einops import rearrange, einsum


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

def gradient_clipping(params: list, max_l2_norm: float, eps: float = 1e-6):
    assert max_l2_norm > 0, f"Max L2 norm should be positive but it is {max_l2_norm}."
    # get global norm
    grads = [param.grad.flatten() for param in params if param.grad is not None]
    l2_norm = ((torch.cat(grads)**2).sum())**0.5
    # check if current norm is too large
    if l2_norm <= max_l2_norm:
        return
    # update gradients
    for param in params:
        if param.grad is None:
            continue
        ratio = max_l2_norm / (l2_norm + eps)
        param.grad *= ratio

def data_loading(x: npt.NDArray, batch_size: int, context_length: int, device: torch.device | None = None) -> (torch.Tensor, torch.Tensor):
    # TODO: clarify notes about np.memmap
    # create masks to sample from numpy
    start_seqs = random.randint(0, x.shape[0] - context_length, size=batch_size)[:, None] # NOTE: change high
    steps_curr = np.arange(context_length)[None, :]
    steps_next = np.arange(1, context_length + 1)[None, :]
    mask_curr, mask_next = start_seqs + steps_curr, start_seqs + steps_next
    # sample numpy tokens
    tokens_curr_np = x[mask_curr]
    tokens_next_np = x[mask_next]
    # convert to PyTorch
    tokens_curr = torch.from_numpy(tokens_curr_np).to(device)
    tokens_next = torch.from_numpy(tokens_next_np).to(device)
    return tokens_curr, tokens_next