import torch
import torch.nn as nn

class Word2VecLoss(nn.Module):
    def __init__(self):
        super(Word2VecLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, positives, negatives):
        return -self.sigmoid(positives).mean()-self.sigmoid(negatives).mean()
