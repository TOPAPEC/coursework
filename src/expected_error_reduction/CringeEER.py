import logging
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import precision_score
from tqdm import tqdm
from modules.models.Linear import LogReg
from modules.models.Wraps import TorchClassifierWrap


class CringeEER:
    @staticmethod
    def expectedErrorReduction(sample, classes, X_train, y_train, base_model, X_pool):
        eer = 0.0
        for cls in classes:
            model = TorchClassifierWrap(LogReg(X_train.shape[1], len(classes)), 100, X_train.shape[1], X_train.shape[0])
            X_append = np.asarray([sample]).reshape((1, X_train.shape[1]))
            y_append = np.asarray([cls]).reshape(1)
            X_new = np.append(X_train, X_append, axis=0)
            y_new = np.append(y_train, y_append, axis=0)
            model.fit(X_new, y_new)
            proba = model.predict_proba(X_pool)
            y_true = model.predict(X_pool)
            loss = float(nn.NLLLoss()(torch.tensor(proba), torch.tensor(y_true, dtype=torch.int64))) / X_pool.shape[0]
            eer += loss * base_model.predict_proba(sample)[0][cls]
        return eer

    @staticmethod
    def EERsampling(X_train, y_train, X_pool, y_pool, X_val, y_val, itr):
        classes = np.unique(list(range(np.max(y_pool) + 1)))
        model = TorchClassifierWrap(LogReg(X_train.shape[1], len(classes)), 100, X_train.shape[1], 10)
        model.fit(X_train, y_train)
        metrics = []
        for iteration in tqdm(range(itr)):
            logging.info(f"Train size is {X_train.shape}")
            sample_ind = np.random.choice(X_pool.shape[0], 10, replace=False)
            loss_reduction = np.ones(sample_ind.shape)
            sampled = X_pool[sample_ind]
            for i, sample in enumerate(sampled):
                loss_reduction[i] = CringeEER.expectedErrorReduction(sample, classes, X_train, y_train, model, X_pool)
            srt = np.argsort(loss_reduction)
            srt = np.flip(srt)
            logging.info(srt[0])
            logging.info(sample_ind[srt[0]])
            X_train = np.append(X_train, X_pool[sample_ind[srt[0]]].reshape(-1, X_pool.shape[1]), axis=0)
            y_train = np.append(y_train, y_pool[sample_ind[srt[0]]].reshape(-1), axis=0)
            X_pool = np.delete(X_pool, sample_ind[srt[0]], axis=0)
            y_pool = np.delete(y_pool, sample_ind[srt[0]], axis=0)
            model = TorchClassifierWrap(LogReg(X_train.shape[1], len(classes)), 100, X_train.shape[1], 10)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            metrics.append(precision_score(y_pred, y_val, average="macro"))
        return metrics

    @staticmethod
    def Randomsampling(X_train, y_train, X_pool, y_pool, X_val, y_val, itr):
        classes = np.unique(list(range(np.max(y_pool) + 1)))
        metrics = []
        for iteration in tqdm(range(itr)):
            logging.info(f"Train size is {X_train.shape}")
            sample_ind = list(np.random.choice(X_pool.shape[0], 10, replace=False))
            loss_reduction = np.ones((len(sample_ind)))
            sampled = X_pool[sample_ind]
            for i, sample in enumerate(sampled):
                loss_reduction[i] = np.random.random_sample()
            srt = np.argsort(loss_reduction)
            X_train = np.append(X_train, X_pool[sample_ind[srt[0]]].reshape(1, -1), axis=0)
            y_train = np.append(y_train, y_pool[sample_ind[srt[0]]].reshape(1), axis=0)
            X_pool = np.delete(X_pool, sample_ind[srt[0]], axis=0)
            y_pool = np.delete(y_pool, sample_ind[srt[0]], axis=0)
            model = TorchClassifierWrap(LogReg(X_train.shape[1], len(classes)), 100, X_train.shape[1], 10)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            metrics.append(precision_score(y_pred, y_val, average="macro"))
        return metrics
