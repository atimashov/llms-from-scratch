import torch
from torch import nn
from .core import Linear
from .nonlinearity import SiLU

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.W1 = Linear(d_model, d_hidden, device = device, dtype=dtype)
        self.W2 = Linear(d_hidden, d_model, device = device, dtype=dtype)
        self.W3 = Linear(d_model, d_hidden, device = device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, X):
        return self.W2(self.silu(self.W1(X)) * self.W3(X))