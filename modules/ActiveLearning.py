import numpy as np
from scipy.stats import entropy
from modules.ActiveLearningBase import ActiveLearningBase


class ConfidenceSamplingSuggestion(ActiveLearningBase):
    def __init__(self, n_top=1000):
        super().__init__(n_top)
        self.n_top = n_top

    def get_samples_for_labeling(self, model, X_test, y_test):
        y_proba = model.predict_proba(X_test)
        y_proba = np.max(y_proba, axis=1)
        ind = np.lexsort((y_test, y_proba))
        return "oracle", ind[:min(self.n_top, y_proba.shape[0])]


class MarginSampling(ActiveLearningBase):
    def __init__(self, n_top=1000):
        super().__init__(n_top)
        self.n_top = n_top

    def get_samples_for_labeling(self, model, X_test, y_test):
        y_proba = model.predict_proba(X_test)
        y_proba = np.sort(y_proba, axis=1)[:, ::-1]
        y_proba = y_proba[:, 0] - y_proba[:, 1]
        ind = np.lexsort((y_test, y_proba))
        return "oracle", ind[:min(self.n_top, y_proba.shape[0])]


class EntropySampling(ActiveLearningBase):
    def __init__(self, n_top=1000):
        super().__init__(n_top)
        self.n_top = n_top

    def get_samples_for_labeling(self, model, X_test, y_test):
        y_proba = model.predict_proba(X_test)
        y_proba = entropy(y_proba, axis=1)
        ind = np.lexsort((y_test, y_proba))[::-1]
        return "oracle", ind[:min(self.n_top, y_proba.shape[0])]

class PseudoLabeling(ActiveLearningBase):
    def __init__(self, n_top=1000):
        super().__init__(n_top)
        self.n_top = n_top

    def get_samples_for_labeling(self, model, X_test, y_test):
        y_proba = model.predict_proba(X_test)
        max_ind = np.argmax(y_proba, axis=1)
        y_proba = np.max(y_proba, axis=1)
        ind = np.lexsort((max_ind, y_proba))[::-1]
        ind_to_return = ind[:min(self.n_top, y_proba.shape[0])]
        return "relabeling", ind_to_return, max_ind[ind_to_return]
