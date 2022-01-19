import numpy as np
from faiss import IndexFlatL2
from modAL.models.base import BaseEstimator
from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax, shuffled_argmax
from scipy.stats import entropy
# import Base


def random_sampling(classifier: BaseEstimator, X: modALinput,
                               n_instances: int = 1,
                               **disputable_points_kwargs):
    indices = np.random.choice(X.shape[0], n_instances, replace=False)
    return indices

# class RandomSampling(Base.ActiveLearningBase):
#     def __init__(self, n_top=1000):
#         super().__init__(n_top)
#         self.n_top = n_top
#
#     def get_samples_for_labeling(self, model, X_test, y_test):
#         ind = np.random.choice(X_test.shape[0], self.n_top, replace=False)
#         return "oracle", ind[:min(self.n_top, y_test.shape[0])]
#
#
# class MarginSampling(Base.ActiveLearningBase):
#     def __init__(self, n_top=1000):
#         super().__init__(n_top)
#         self.n_top = n_top
#
#     def get_samples_for_labeling(self, model, X_test, y_test):
#         y_proba = model.predict_proba(X_test)
#         y_proba = np.sort(y_proba, axis=1)[:, ::-1]
#         y_proba = y_proba[:, 0] - y_proba[:, 1]
#         ind = np.lexsort((y_test, y_proba))
#         return "oracle", ind[:min(self.n_top, y_proba.shape[0])]
#
#
# class EntropySampling(Base.ActiveLearningBase):
#     def __init__(self, n_top=1000):
#         super().__init__(n_top)
#         self.n_top = n_top
#
#     def get_samples_for_labeling(self, model, X_test, y_test):
#         y_proba = model.predict_proba(X_test)
#         y_proba = entropy(y_proba, axis=1)
#         ind = np.lexsort((y_test, y_proba))[::-1]
#         return "oracle", ind[:min(self.n_top, y_proba.shape[0])]
#
#
# class ConfidenceSampling(Base.ActiveLearningBase):
#     def __init__(self, n_top=1000):
#         super().__init__(n_top)
#         self.n_top = n_top
#
#     def get_samples_for_labeling(self, model, X_test, y_test):
#         y_proba = model.predict_proba(X_test)
#         y_proba = np.max(y_proba, axis=1)
#         ind = np.lexsort((y_test, y_proba))
#         return "oracle", ind[:min(self.n_top, y_proba.shape[0])]
