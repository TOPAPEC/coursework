import numpy as np


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
