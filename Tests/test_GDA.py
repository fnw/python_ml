import unittest

import numpy as np

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LR

from python_ml.Linear.GaussianDiscriminantAnalysis import GDAClassifier


class TestGDA(unittest.TestCase):
    def setUp(self):
        n_samples = 1000
        percent_train = 0.75

        n_train = int(percent_train * n_samples)

        np.random.seed(987321)

        X, y = make_classification(n_samples=n_samples, n_features=3, n_informative=2, n_repeated=0, n_redundant=0)

        self.train_X, self.train_y = X[:n_train, :], y[:n_train]
        self.test_X, self.test_y = X[n_train:, :], y[n_train:]

        self.clf = GDAClassifier()

        self.clf.fit(self.train_X, self.train_y)

        np.random.seed(None)

    def test_means(self):
        self.assertTrue(self.clf.means is not None)

    def test_vars(self):
        self.assertTrue(self.clf.vars is not None)

    def test_accuracy(self):
        y_pred = self.clf.predict(self.test_X)

        acc_clf = accuracy_score(y_pred, self.test_y)

        sk = LR()
        sk.fit(self.train_X, self.train_y)

        y_pred = sk.predict(self.test_X)

        acc_sk = accuracy_score(self.test_y, y_pred)

        min_acc, max_acc = min(acc_clf, acc_sk), min(acc_clf, acc_sk)

        percent_difference = (max_acc - min_acc) / max_acc

        self.assertLess(percent_difference, 0.01)


if __name__ == '__main__':
    unittest.main()
