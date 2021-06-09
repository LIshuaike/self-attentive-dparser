import torch
import torch.nn as nn


class HLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        b = x * torch.log(x)
        b = -1.0 * b.sum()
        return b