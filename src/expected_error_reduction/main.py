import logging

from sklearn.metrics import precision_score, recall_score, accuracy_score

import CringeEER
import SlowEER
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from modules.models.Linear import LogReg
from modules.models.Wraps import TorchClassifierWrap


def main():
    # https://archive.ics.uci.edu/ml/datasets/Wine+Quality - red wine quality
    raw_dataset = pd.read_csv("../../tmp/datasets/abalone.data", sep=",", header=None)
    print(raw_dataset)

    X = np.asarray(raw_dataset.drop([0, 8], axis=1))
    y = np.asarray(raw_dataset.iloc[:, 8])
    # print(OneHotEncoder(sparse=False).fit_transform(X=raw_dataset[0].to_numpy().reshape(-1, 1)))
    X = np.hstack((OneHotEncoder(sparse=False).fit_transform(X=raw_dataset[0].to_numpy().reshape(-1, 1)), X))
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
    X_pool, X_val, y_pool, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=42)

    # classes = np.unique(list(range(np.max(y_pool) + 1)))
    # print(X_train.shape)
    logging.info(f"EERampling {str(CringeEER.CringeEER.EERsampling(X_train, y_train, X_pool, y_pool, X_val, y_val, 10))}")

    logging.info(f"Random sampling {str(CringeEER.CringeEER.Randomsampling(X_train, y_train, X_pool, y_pool, X_val, y_val, 10))}")
    # model = TorchClassifierWrap(LogReg(X_train.shape[1], len(classes)), 1000, X_train.shape[1], 100)
    # model.fit(X_train, y_train)
    # pred = model.predict(X_pool)
    # logging.info(accuracy_score(pred, y_pool))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()
