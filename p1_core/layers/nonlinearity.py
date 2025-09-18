import torch
from torch import  nn
import math

class ReLU(nn.Module):
    def forward(self, x):
        return torch.where(x < 0, torch.zeros_like(x), x)
    
class LeakyReLU(nn.Module):
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return torch.where(x < 0, self.alpha * x, x)

class SqReLU(nn.Module):
    def forward(self, x):
        return torch.where(x < 0, torch.zeros_like(x), x ** 2)

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/ math.pi) * (x + 0.044715 * x ** 3)))