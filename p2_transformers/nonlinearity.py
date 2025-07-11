import torch
from torch import  nn

class ReLU(nn.Module):
    def forward(self, x):
        return torch.where(x < 0, torch.zeros_like(x), x)
    
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)