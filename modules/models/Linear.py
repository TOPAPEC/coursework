import torch.nn as nn


class LogReg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.seq(x)