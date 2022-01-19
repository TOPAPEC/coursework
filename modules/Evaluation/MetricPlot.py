import pandas as pd
import numpy as np
import matplotlib as mpl

from typing import Tuple, List
from matplotlib import pyplot as plt
from modules.models.Wraps import ModelWrap
from modAL.models import ActiveLearner
from typing import Union
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_metrics(model: Union[ModelWrap, ActiveLearner], X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    y_pred = model.predict(X_test)
    metrics = pd.DataFrame({"accuracy": [accuracy_score(y_test, y_pred)],
                            "f1_score": [f1_score(y_test, y_pred, average="macro")],
                            "precision_score": [precision_score(y_test, y_pred, average="macro")],
                            "recall_score": [recall_score(y_test, y_pred, average="macro")]})
    return metrics


def _transform_metrics(labels: List[str], metrics: List[pd.DataFrame]) -> Tuple[List[str], List[pd.DataFrame]]:
    metric_labels = []
    metric_dfs = []
    for metric in metrics[0].columns:
        dataframe = pd.DataFrame()
        for method_name, method_metrics in zip(labels, metrics):
            dataframe[method_name] = method_metrics[metric]
        metric_labels.append(metric)
        metric_dfs.append(dataframe.reset_index(drop=True))
    return metric_labels, metric_dfs


def plot_metrics(labels_metrics: Tuple[List[str], List[pd.DataFrame]], plot_size=(9.5, 7)):
    mpl.style.use('ggplot')
    labels, metrics = labels_metrics
    desired_shape = metrics[0].shape
    assert (len(labels) == len(metrics))
    for metric in metrics:
        assert metric.shape == desired_shape
    plot_number = len(labels)
    fig_size = (plot_size[0] * plot_number, plot_size[1])
    fig, axs = plt.subplots(ncols=plot_number, figsize=fig_size)
    (metric_labels, metric_dfs) = _transform_metrics(labels, metrics)
    for metric_label, metric_df, ax in zip(metric_labels, metric_dfs, axs):
        metric_df.plot(ax=ax, title=metric_label)
