import numpy as np
from modules.ActiveLearning import ActiveLearningBase
from faiss import IndexFlatL2
from modAL.models.base import BaseEstimator
from modAL.utils.data import modALinput
from functools import partial
from scipy.stats import entropy


def _nearest_neighbours_to_entropy(nearest_neighbours: np.ndarray, min_bins: int):
    bin_count = np.apply_along_axis(partial(np.bincount, minlength=min_bins), 1, nearest_neighbours)
    return entropy(bin_count, axis=1)


def classifier_train_confusion(classifier: BaseEstimator, X: modALinput,
                               index: IndexFlatL2, n_nearest: int = 100,
                               n_instances: int = 1, random_tie_break: bool = False,
                               **uncertainty_measure_kwargs):
    dist, ind = index.search(X, n_nearest)
    entropies = _nearest_neighbours_to_entropy(classifier.y_training[ind], np.unique(classifier.y_training).shape[0])
    indices_to_return = np.argsort(entropies)[::-1]
    return indices_to_return[:n_instances]


def classifier_pseudolabeling(classifier: BaseEstimator, X: modALinput,
                              n_nearest: int = 100,
                              n_instances: int = 1, random_tie_break: bool = False,
                              **uncertainty_measure_kwargs):
    y_proba = classifier.predict_proba(X)
    entropies = entropy(y_proba, axis=1)
    ind = np.argsort(entropies)
    return ind[:n_instances]