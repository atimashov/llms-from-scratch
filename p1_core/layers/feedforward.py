import torch
from torch import nn
from .core import Linear
from .nonlinearity import ReLU, LeakyReLU, SqReLU, SiLU, GELU

name_to_func = {
    "ReLU": ReLU,
    "LeakyReLU": LeakyReLU,
    "SqReLU": SqReLU,
    "SiLU": SiLU,
    "GELU": GELU
}

class GatedFFN(nn.Module):
    def __init__(
        self, d_model: int, d_hidden: int, init_type: str = 'xavier', clip_w: float = 3.0, 
        activation: str = SiLU, device: torch.device | None = None, dtype: torch.dtype | None = None
        ):
        super().__init__()
        self.W1 = Linear(d_model, d_hidden, init_type, clip_w, device = device, dtype=dtype)
        self.W2 = Linear(d_hidden, d_model, init_type, clip_w, device = device, dtype=dtype)
        self.V = Linear(d_model, d_hidden, init_type, clip_w, device = device, dtype=dtype)
        self.activation = name_to_func[activation]()

    def forward(self, X):
        return self.W2(self.activation(self.W1(X)) * self.V(X))

class FFN(nn.Module):
    def __init__(
        self, d_model: int, d_hidden: int, init_type: str = 'xavier', clip_w: float = 3.0, 
        activation: str = "SiLU", device: torch.device | None = None, dtype: torch.dtype | None = None
        ):
        super().__init__()
        self.W1 = Linear(d_model, d_hidden, init_type, clip_w, device = device, dtype=dtype)
        self.W2 = Linear(d_hidden, d_model, init_type, clip_w, device = device, dtype=dtype)
        self.activation = name_to_func[activation]()

    def forward(self, X):
        return self.W2(self.activation(self.W1(X)))