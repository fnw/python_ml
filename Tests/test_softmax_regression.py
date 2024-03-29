import unittest

import numpy as np

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import LabelBinarizer

from python_ml.Linear.SoftmaxRegression import SoftmaxRegressionClassifier


class TestSoftmaxRegression(unittest.TestCase):
    def setUp(self):
        n_samples = 100
        percent_train = 0.75

        n_train = int(percent_train * n_samples)

        np.random.seed(987321)

        X, y = make_classification(n_samples=n_samples, n_classes=3, n_features=6, n_informative=6, n_repeated=0,
                                   n_redundant=0)

        encoder = LabelBinarizer()

        y_one_hot = encoder.fit_transform(y)

        self.train_X, self.train_y_one_hot, self.train_y = X[:n_train, :], y_one_hot[:n_train, :], y[:n_train]
        self.test_X, self.test_y = X[n_train:, :], y[n_train:]

        self.clf = SoftmaxRegressionClassifier()

        self.clf.fit(self.train_X, self.train_y_one_hot)

        np.random.seed(None)

    def test_was_fit(self):
        self.assertTrue(self.clf.weights is not None)

    def test_accuracy(self):
        y_pred = self.clf.predict(self.test_X)

        acc_clf = accuracy_score(y_pred, self.test_y)

        sk = LR()
        sk.fit(self.train_X, self.train_y)

        y_pred = sk.predict(self.test_X)

        acc_sk = accuracy_score(self.test_y, y_pred)

        min_acc, max_acc = min(acc_clf, acc_sk), max(acc_clf, acc_sk)

        percent_difference = (max_acc - min_acc) / max_acc

        self.assertLess(percent_difference, 0.01)


if __name__ == '__main__':
    unittest.main()
