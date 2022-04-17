import functools
import logging

import faiss
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from modAL import Committee
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from modules.models.Linear import LogReg
from modules.models.Neuro import Neuro
from modules.models.Wraps import TorchClassifierWrap
from modAL.models import ActiveLearner
from sklearn.svm import LinearSVC
from modules.ActiveLearning.Heuristics import disputable_points
from modules.ActiveLearning.Samplings import random_sampling

from modAL.uncertainty import classifier_uncertainty, classifier_margin, classifier_entropy
from modAL.density import information_density
from modAL.disagreement import vote_entropy, vote_entropy_sampling


def main():
    raw_dataset = pd.read_csv("../../tmp/datasets/heart.csv", sep=",")
    print(raw_dataset)

    X = np.asarray(raw_dataset.drop(["output"], axis=1))
    y = np.asarray(raw_dataset.loc[:, "output"])
    # print(OneHotEncoder(sparse=False).fit_transform(X=raw_dataset[0].to_numpy().reshape(-1, 1)))
    # X = np.hstack((OneHotEncoder(sparse=False).fit_transform(X=raw_dataset[0].to_numpy().reshape(-1, 1)), X))
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)
    X_pool, X_val, y_pool, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=42)
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_pool = X_pool.astype(np.float32)
    classes = np.unique(y)
    print(X_train.shape, y_train.shape)
    # model = TorchClassifierWrap(Neuro([X_train.shape[1], X_train.shape[1] // 2, len(classes)]), 1000, X_train.shape[1], 100)
    model = TorchClassifierWrap(LogReg(X_train.shape[1], len(classes)), 1000, X_train.shape[1], 100)
    # model = CatBoostClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(precision_score(y_pred, y_val, average="macro"))
    index = faiss.IndexFlatIP(X_train.shape[1])
    index.add(X_train)
    assert index.is_trained
    committee = Committee(
        learner_list=[
            ActiveLearner(
                estimator=TorchClassifierWrap(LogReg(X_train.shape[1], len(classes)), 1000, X_train.shape[1], 100),
                X_train=X_train, y_train=y_train
            ),
            ActiveLearner(
                estimator=LinearSVC(),
                X_train=X_train, y_train=y_train
            ),
            ActiveLearner(
                estimator=TorchClassifierWrap(Neuro([X_train.shape[1], X_train.shape[1] // 2, len(classes)]), 1000,
                                              X_train.shape[1], 100),
                X_train=X_train, y_train=y_train
            )
        ],
        query_strategy=vote_entropy_sampling
    )
    samplers = [
        functools.partial(classifier_margin, classifier=model),
        functools.partial(classifier_uncertainty, classifier=model),
        functools.partial(classifier_entropy, classifier=model),
        functools.partial(disputable_points, index=index,
                          classifier=ActiveLearner(model, X_training=X_train, y_training=y_train)),
        functools.partial(information_density, metric="cosine"),
        functools.partial(vote_entropy, committee=committee)
    ]
    for itr in range(10):
        rank = sampler(X_pool, samplers)
        X_train.append(rank[:10])

def sampler(X_pool, samplers):
    ranks = np.zeros((len(samplers), X_pool.shape[0]), dtype=np.int64)
    information_density(X_pool, metric="cosine")
    for i, spr in enumerate(samplers):
        if (i == 0):
        query_idx = np.flip(np.argsort(spr(X=X_pool)))
        ranks[i] = query_idx
    print(ranks.shape)
    merged_rank = np.zeros(ranks.shape[1])
    for sample in range(X_pool.shape[0]):
        for arr in ranks:
            merged_rank[sample] += arr[sample] ** 2
    final_rank = np.argsort(merged_rank)
    return final_rank


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
