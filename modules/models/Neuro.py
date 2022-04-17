import torch.nn as nn


class Neuro(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for left, right in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(left, right))
            layers.append(nn.ReLU())
        self.seq = nn.Sequential(
            *layers,
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.seq(x)