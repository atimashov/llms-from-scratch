import torch
import torch.nn as nn

class Word2VecLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, positives, negatives):
        return -self.sigmoid(positives).mean()-self.sigmoid(negatives).mean()
    
class GloveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weights, deltas):
        return (weights * deltas**2).sum()
