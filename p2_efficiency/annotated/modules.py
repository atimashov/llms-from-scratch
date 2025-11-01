__all__ = [
    "annotated_scaled_dot_product_attention"
]

from einops import einsum
import torch
import torch.cuda.nvtx as nvtx
from p1_core.utils import softmax


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None):
    """
    Q, K:  (batch_size, ..., seq_len, d_k)
    V:  (batch_size, ..., seq_len, d_v)
    """
    d_k = K.shape[-1]
    # seq_len is the same for both, but I distinguish the ordering
    with nvtx.range("compute attention scores"):
        scores = einsum(Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") 
        scores = scores.clamp(min = -80, max=80.0)
    with nvtx.range("apply mask"):
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
    with nvtx.range("computing softmax"):
        weights = softmax(scores / (d_k ** 0.5), dim = -1)
    with nvtx.range("computing output projection"):
        att = einsum(weights, V, "... seq_len seq_len2, ... seq_len2 d_v -> ... seq_len d_v")
    return att