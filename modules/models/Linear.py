from modules.models import ModelWrap
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import random
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(300, 1013),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.seq(x)


class LinearModelTorch(ModelWrap):
    def __init__(self, model, itr, random_seed=42):
        self.random_seed = random_seed
        self.model = model
        self.last_loss = 0
        self.iter = itr

    def fit(self, X, y):
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        torch.use_deterministic_algorithms(False)
        random.seed(self.random_seed)
        torch.cuda.empty_cache()
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.int64))
        self.model = self.model.cuda()
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=1e-5)
        criterion = nn.NLLLoss()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=10000, num_workers=0)
        for epoch in tqdm(range(self.iter)):
            for bX, by in dataloader:
                bX = bX.view(-1, 300)
                bX = bX.cuda()
                by = by.cuda()
                optimizer.zero_grad()
                output = self.model(bX)
                loss = criterion(output, by)
                loss.backward()
                optimizer.step()
            #             if (epoch % 10 == 0):
            #                 print(f"ep{epoch}: {loss}")
            torch.cuda.empty_cache()
        X = X.cpu().detach().numpy()
        y = y.cpu().detach().numpy()

    def predict(self, X):
        X = torch.from_numpy(X)
        X = X.view(-1, 300)
        X = X.cpu()
        self.model = self.model.cpu()
        self.model = self.model.eval()
        result = torch.argmax(self.model(X.float()), axis=1)
        X = X.detach().numpy()
        return result.detach().numpy()

    def predict_proba(self, X):
        X = torch.from_numpy(X)
        X = X.view(-1, 300)
        X = X.cpu()
        self.model.cpu()
        result = torch.exp(self.model(X.float()))
        X = X.detach().numpy()
        return result.detach().numpy()
