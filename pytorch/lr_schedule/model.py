import torch
from torch import nn
from schedules import LinearWarmupExponentialDecay


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 2),
            nn.Linear(2, 8),
            nn.Linear(8, 4),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        x = self.layer(x)
        return x
