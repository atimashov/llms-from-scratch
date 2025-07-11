import torch
from torch import  nn

class LayerNorm(nn.Module):
    """
    Process an input tensor of shape (batch_size, sequence_length, d_model) 
    Returns a tensor of the same shape
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))
        self.beta = nn.Parameter(torch.zeros(d_model, device = device, dtype = dtype))
        self.eps = eps

    def forward(self, x):
        # x shape: (batch, dim)
        mu = x.mean(dim = -1, keepdim = True) 
        sigma_sq = ((x - mu) ** 2).mean(dim = -1, keepdim = True) 
        x_hat = (x - mu) / torch.sqrt(sigma_sq + self.eps)
        out = self.gamma * x_hat + self.beta
        return out
        
class RMSNorm(nn.Module):
    """
    Process an input tensor of shape (batch_size, sequence_length, d_model) 
    Returns a tensor of the same shape
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, dim)
        rms = torch.sqrt((x ** 2).mean(dim = -1, keepdim = True) + self.eps) 
        out = x / rms * self.gamma 
        return out