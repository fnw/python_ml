import numpy as np
from scipy.stats import mode

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import make_classification, load_boston


class KNearestNeighbors:
    def __init__(self, k=None):
        self.X = None
        self.y = None
        self.mode = None
        self.k = k

    def fit(self, X, y, mode='brute'):
        if mode == 'kdtree':
            raise NotImplementedError('Mode kdtree is not yet implemented')
        elif mode == 'brute':
            self.mode = 'brute'
        else:
            raise ValueError('Invalid mode: ', mode)

        self.X = X
        self.y = y

        return self

    def get_k_nearest_neighbors(self, X, k=None, return_idx=False, return_distances=False):
        if k is None:
            if self.k is None:
                raise ValueError('You must specify the number of neighbors')
            else:
                k = self.k

        dists = np.sum((X[:, np.newaxis, :] - self.X) ** 2, axis=-1)

        idx = np.argpartition(dists, range(k), axis=1)[:, :k]

        nn = self.X[idx, :]

        if return_idx:
            if return_distances:
                return nn, idx, np.take_along_axis(dists, idx, 1)
            else:
                return nn, idx
        elif return_distances:
            return nn, np.take_along_axis(dists, idx, 1)
        else:
            return nn


class KNearestNeighborsClassifier(KNearestNeighbors):
    def predict(self, X):
        _, idx = self.get_k_nearest_neighbors(X, return_idx=True)

        y_nn = self.y[idx]

        y_pred = mode(y_nn, axis=1)[0]

        return y_pred.squeeze()


class KNearestNeighborsRegressor(KNearestNeighbors):
    def predict(self, X):
        _, idx = self.get_k_nearest_neighbors(X, return_idx=True)

        y_nn = self.y[idx]

        y_pred = np.mean(y_nn, axis=1)

        return y_pred