import torch
import torch.nn as nn


class limiterActivation(nn.Module):
    def __init__(self, min_val, max_val):
        super(limiterActivation, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, min=self.min_val, max=self.max_val)
