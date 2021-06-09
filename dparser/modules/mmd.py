import torch
import torch.nn as nn
from functools import partial

from dparser.utils import gaussian_kernel_matrix, maximum_mean_discrepancy


class MMD(nn.Module):
    def __init__(self, encoder, configs):
        super().__init__()
        self.sigmas = torch.FloatTensor([
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
            1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ])
        self.gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas=self.sigmas)

    def forward(self, hs, ht):
        loss_value = maximum_mean_discrepancy(
            hs, ht, kernel=self.gaussian_kernel)
        return torch.clamp(loss_value, min=1e-4)
