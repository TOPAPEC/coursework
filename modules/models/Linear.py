import torch.nn as nn


class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(300, 1013),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.seq(x)