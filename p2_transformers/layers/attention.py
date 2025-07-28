from einops import rearrange, einsum
import torch
from torch import nn
from .pos_enc import RoPE
from utils import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    """
    d_model: int Dimensionality of the Transformer block inputs
    num_heads: int Number of heads to use in multi-head self-attention.
    TODO: check dtype/device correctness; add documentation; work on reproducibility (seed)
    """
    def __init__(self, d_model: int, num_heads: int, theta: float = 10000.0, context_length = 10000, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model, self.num_heads = d_model, num_heads
        self.d_k, self.d_v = d_model // num_heads, d_model // num_heads
        # init matrices
        std = 2 / (num_heads * self.d_k + d_model)
        data = torch.empty(num_heads * self.d_k, d_model, dtype=dtype, device=device)
        nn.init.trunc_normal_(data, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.P_Q = nn.Parameter(data)
        data = torch.empty(num_heads * self.d_k, d_model, dtype=dtype, device=device)
        nn.init.trunc_normal_(data, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.P_K = nn.Parameter(data)
        std = 2 / (num_heads * self.d_v + d_model)
        data = torch.empty(num_heads * self.d_v, d_model, dtype=dtype, device=device)
        nn.init.trunc_normal_(data, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.P_V = nn.Parameter(data)
        data = torch.empty(d_model, num_heads * self.d_v, dtype=dtype, device=device)
        nn.init.trunc_normal_(data, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.P_O = nn.Parameter(data)
        # init RoPE
        self.rope = RoPE(theta = theta, d_k = self.d_k, max_seq_len= context_length, device=device, dtype = dtype)
        self.device = device

    def forward(self, x: torch.Tensor, is_masked: bool = True, with_rope = True, token_positions = None):
        # project x to get queries, keys and values
        Q = einsum(x, self.P_Q, "... d, hd_k d -> ... hd_k") # unplug
        Q = rearrange(Q, "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads)
        K = einsum(x, self.P_K, "... d, hd_k d -> ... hd_k")
        K = rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads)
        V = einsum(x, self.P_V, "... d, hd_v d -> ... hd_v")
        V = rearrange(V, "...  seq_len (h d_v) -> ... h seq_len d_v", h = self.num_heads)
        # apply RoPE
        if with_rope:
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
        O = einsum(scaled_mh_att, self.P_O, "... seq_len hd_v, d hd_v -> ... seq_len d")
        return O