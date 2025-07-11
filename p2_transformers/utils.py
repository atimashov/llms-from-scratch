import torch
from torch import nn
from einops import rearrange, einsum


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert 0 <= dim < x.dim()
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
    att = einsum(weights, V, "... seq_len seq_len2, ... seq_len2 d_v -> ... seq_len, d_v")
    return att
