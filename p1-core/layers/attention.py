from einops import rearrange, einsum
import torch
from torch import nn
from .pos_enc import RoPE
from .core import Linear
from utils import softmax

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None):
    """
    Q, K:  (batch_size, ..., seq_len, d_k)
    V:  (batch_size, ..., seq_len, d_v)
    """
    d_k = K.shape[-1]
    # seq_len is the same for both, but I distinguish the ordering
    scores = einsum(Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") 
    scores = scores.clamp(min = -80, max=80.0)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    weights = softmax(scores / (d_k ** 0.5), dim = -1)
    att = einsum(weights, V, "... seq_len seq_len2, ... seq_len2 d_v -> ... seq_len d_v")
    return att

class MultiHeadSelfAttention(nn.Module):
    """
    d_model: int Dimensionality of the Transformer block inputs
    num_heads: int Number of heads to use in multi-head self-attention.
    TODO: check dtype/device correctness; add documentation; work on reproducibility (seed)
    """
    def __init__(
        self, d_model: int, num_heads: int, theta: float = 10000.0, context_length = 10000, 
        init_type: str = 'xavier', clip_w: float = 3.0, device: torch.device | None = None, dtype: torch.dtype | None = None
        ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model, self.num_heads = d_model, num_heads
        self.d_k, self.d_v = d_model // num_heads, d_model // num_heads
        # init matrices
        self.P_Q = Linear(d_model, num_heads * self.d_k, init_type, clip_w, device = device, dtype=dtype)
        self.P_K = Linear(d_model, num_heads * self.d_k, init_type, clip_w, device = device, dtype=dtype)
        self.P_V = Linear(d_model, num_heads * self.d_v, init_type, clip_w, device = device, dtype=dtype)
        self.P_O = Linear(num_heads * self.d_v, d_model, init_type, clip_w, device = device, dtype=dtype)
        # init RoPE
        self.rope = RoPE(theta = theta, d_k = self.d_k, max_seq_len= context_length, device=device, dtype = dtype) if theta is not None else None
        self.device = device

    def forward(self, x: torch.Tensor, is_masked: bool = True, with_rope = True, token_positions = None):
        # project x to get queries, keys and values
        Q = self.P_Q(x)
        Q = rearrange(Q, "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads)
        K = self.P_K(x)
        K = rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads)
        V = self.P_V(x)
        V = rearrange(V, "...  seq_len (h d_v) -> ... h seq_len d_v", h = self.num_heads)
        # apply RoPE
        if with_rope and self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)
        # create mask
        if is_masked:
            mask = ~torch.triu(torch.full((Q.shape[-2], K.shape[-2]), True, device = self.device), diagonal=1)
        else:
            mask = None
        # calculate scaled attention
        scaled_mh_att = scaled_dot_product_attention(Q, K, V, mask)
        scaled_mh_att = rearrange(scaled_mh_att, "... h seq_len d_v -> ... seq_len (h d_v)")
        # project on output
        O = self.P_O(scaled_mh_att) # O = einsum(scaled_mh_att, self.P_O, "... seq_len hd_v, d hd_v -> ... seq_len d")
        return O