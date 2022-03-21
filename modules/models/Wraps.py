import numpy as np
import torch
import random
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator


class ModelWrap:
    def __init__(self):
        self.model = None
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelWrap':
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass


class TorchClassifierWrap(ModelWrap, BaseEstimator):
    def __init__(self, model: nn.Module, itr: int, input_dimension: int, batch_size: int, random_state: int = 42,
                 verbose: bool = False):
        super().__init__()
        self.random_seed = random_state
        self.model = model
        self.last_loss = 0
        self.iter = itr
        self.verbose = verbose
        self.input_dimension = input_dimension
        self.batch_size = batch_size

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
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=0)
        for epoch in tqdm(range(self.iter)):
            for bX, by in dataloader:
                bX = bX.view(-1, self.input_dimension)
                bX = bX.cuda()
                by = by.cuda()
                optimizer.zero_grad()
                output = self.model(bX)
                loss = criterion(output, by)
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0 and self.verbose:
                print(f"ep{epoch}: {loss}")
            torch.cuda.empty_cache()
        X = X.cpu().detach().numpy()
        y = y.cpu().detach().numpy()

    def predict(self, X, **predict_kwargs):
        X = torch.from_numpy(X)
        X = X.view(-1, self.input_dimension)
        X = X.cpu()
        self.model = self.model.cpu()
        self.model = self.model.eval()
        result = torch.argmax(self.model(X.float()), axis=1)
        X = X.detach().numpy()
        return result.detach().numpy()

    def predict_proba(self, X, **predict_proba_kwargs):
        X = torch.from_numpy(X)
        X = X.view(-1, self.input_dimension)
        X = X.cpu()
        self.model.cpu()
        self.model.eval()
        result = torch.exp(self.model(X.float()))
        X = X.detach().numpy()
        return result.detach().numpy()

    def set_state(self, state):
        self.model.load_state_dict(state)

    def get_state(self):
        return self.model.state_dict()
