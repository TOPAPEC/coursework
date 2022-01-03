import numpy as np
from scipy.stats import entropy


class ActiveLearningBase:
    def __init__(self, n_top=1000):
        self.n_top = n_top

    def get_samples_for_labeling(self, model, X_test, y_test):
        pass





