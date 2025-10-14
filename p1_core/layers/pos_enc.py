import torch
from torch import nn
from einops import rearrange, einsum

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        # TODO: check if dtype or device is None
        assert d_k % 2 ==0
        self.d_k = d_k
        # rotate over even indices        
        position = torch.arange(max_seq_len, dtype=dtype, device=device)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype = dtype, device = device) / d_k))
        emb = einsum(position, inv_freq, "max_seq_len, half_d_k -> max_seq_len half_d_k")
        # register sin and cos  
        self.register_buffer("sin", torch.sin(emb), persistent=False)
        self.register_buffer("cos", torch.cos(emb), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        assert x.shape[-1] == self.d_k
        # choose right positions
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2])
            token_positions = token_positions.to(self.sin.device)
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]
        # split x into even and odd dimensions
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        # apply rotations (it will broadcasr automatically since last 2 dims match)
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos
        # calculate output
        x_out = torch.empty_like(x) # torch.zeros_like(x) is slower in theory
        x_out[..., 0::2] = rot_x1
        x_out[..., 1::2] = rot_x2
        return x_out