import unittest

import numpy as np

from sklearn.datasets import make_classification, load_boston
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from python_ml.NearestNeighbors.knn import KNearestNeighborsRegressor, KNearestNeighborsClassifier


class TestKNN(unittest.TestCase):
    def test_classification(self):
        n_examples = 500
        n_train = int(0.75 * n_examples)

        np.random.seed(987321)
        X, y = make_classification(n_samples=n_examples, n_classes=3, n_features=8, n_informative=8, n_repeated=0,
                                   n_redundant=0)

        train_X, train_y = X[:n_train, :], y[:n_train]
        test_X, test_y = X[n_train:, :], y[n_train:]

        np.random.seed(None)

        # sklearn

        sk = KNeighborsClassifier(n_neighbors=3)
        sk.fit(train_X, train_y)

        dists_sk, idx_sk = sk.kneighbors(test_X)
        y_pred_sk = sk.predict(test_X)

        # Mine

        myknn = KNearestNeighborsClassifier(k=3)

        myknn.fit(train_X, train_y)

        nearest_neighbors_mine, idx, dists = myknn.get_k_nearest_neighbors(test_X, return_idx=True,
                                                                           return_distances=True)

        y_pred = myknn.predict(test_X)

        # I don't take the square root
        self.assertTrue(np.allclose(dists_sk, np.sqrt(dists)))
        self.assertTrue(np.all(idx == idx_sk))
        self.assertTrue(np.all(y_pred == y_pred_sk))

    def test_regression(self):
        X, y = load_boston(return_X_y=True)

        n_examples = len(y)
        n_train = int(0.75 * n_examples)

        np.random.seed(987321)

        train_X, train_y = X[:n_train, :], y[:n_train]
        test_X, test_y = X[n_train:, :], y[n_train:]

        np.random.seed(None)

        # sklearn
        sk = KNeighborsRegressor(n_neighbors=3)
        sk.fit(train_X, train_y)

        dists_sk, idx_sk = sk.kneighbors(test_X)
        y_pred_sk = sk.predict(test_X)

        # Mine
        myknn = KNearestNeighborsRegressor(k=3)

        myknn.fit(train_X, train_y)

        nearest_neighbors_mine, idx, dists = myknn.get_k_nearest_neighbors(test_X, return_idx=True,
                                                                           return_distances=True)

        y_pred = myknn.predict(test_X)

        self.assertTrue(np.allclose(dists_sk, np.sqrt(dists)))
        self.assertTrue(np.all(idx_sk == idx))
        self.assertTrue(np.allclose(y_pred, y_pred_sk))


if __name__ == '__main__':
    unittest.main()
