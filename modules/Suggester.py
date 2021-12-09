import numpy as np
import logging
from modules.ActiveLearningBase import ActiveLearningBase
from modules.ModelWrap import ModelWrap
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from modules.Suggest import Suggest


class Suggester:
    def __init__(self, X, y, random_seed=42, test_size=0.8, label_dict=None):
        self.test_size = test_size
        self.random_seed = random_seed
        self.last_suggest = None
        self.loss_history = []
        self.label_dict = label_dict
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=self.test_size, random_state=random_seed)

    def active_learning_suggest(self, suggest_algorithm: ActiveLearningBase, model: ModelWrap, apply: bool = False,
                                sample_ratio=1):
        num_of_elements = int(self.X_test.shape[0] * sample_ratio)
        sample_ind = np.random.choice(self.X_test.shape[0], num_of_elements, replace=False)
        logging.info(f"Looking at {num_of_elements} samples from test.")
        suggest_output = \
            suggest_algorithm.get_samples_for_labeling(model, self.X_test[sample_ind, :], self.y_test[sample_ind])
        if suggest_output[0] == "oracle":
            ind = sample_ind[suggest_output[1]]
            self.last_suggest = Suggest(indices=ind, from_to="oracle")
        elif suggest_output[0] == "labeling":
            ind = sample_ind[suggest_output[1]]
            labels = suggest_output[2]
            self.last_suggest = Suggest(indices=ind, labels=labels, from_to="labeling")

    def train_suggest(self, suggest_algorithm: ActiveLearningBase, model: ModelWrap, apply: bool = False,
                                sample_ratio=1):
        num_of_elements = int(self.X_train.shape[0] * sample_ratio)
        sample_ind = np.random.choice(self.X_train.shape[0], num_of_elements, replace=False)
        logging.info(f"Looking at {num_of_elements} samples from train.")
        suggest_output = \
            suggest_algorithm.get_samples_for_labeling(model, self.X_train[sample_ind, :], self.y_train[sample_ind])
        if suggest_output[0] == "relabeling":
            pass


    # Suggest should be passed here, probably will make it an object
    def apply_last_suggest(self):
        if self.last_suggest is None:
            logging.warning("There is no new unapplied suggests available. Aborting.")
            return
        if self.last_suggest.from_to == "oracle":
            ind = self.last_suggest.indices
            self.X_train = np.append(self.X_train, self.X_test[ind], axis=0)
            self.y_train = np.append(self.y_train, self.y_test[ind], axis=0)
            self.X_test = np.delete(self.X_test, ind, axis=0)
            self.y_test = np.delete(self.y_test, ind, axis=0)
        elif self.last_suggest.from_to == "labeling":
            ind = self.last_suggest.indices
            labels = self.last_suggest.labels
            self.X_train = np.append(self.X_train, self.X_test[ind], axis=0)
            self.y_train = np.append(self.y_train, labels, axis=0)
            self.X_test = np.delete(self.X_test, ind, axis=0)
            self.y_test = np.delete(self.y_test, ind, axis=0)
        else:
            pass
        self.last_suggest = None

    def print_last_suggest(self):
        pass

    def get_loss_history(self):
        pass

    # In future it should take dict of "metricname"-metricalg
    def evaluate_metrics(self, model: ModelWrap) -> dict:
        y_pred = model.predict(self.X_test)
        metrics = {}
        metrics["accuracy"] = accuracy_score(self.y_test, y_pred)
        metrics["f1_score"] = f1_score(self.y_test, y_pred, average="macro")
        metrics["precision_score"] = precision_score(self.y_test, y_pred, average="macro")
        metrics["recall_score"] = recall_score(self.y_test, y_pred, average="macro")
        return metrics

    def get_metric_history(self):
        pass
