import unittest

import numpy as np

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from python_ml.Tree.DecisionTree import DecisionTree


class TestLogisticRegression(unittest.TestCase):
    def setUp(self) -> None:
        n_examples = 500
        n_train = int(0.75 * n_examples)

        np.random.seed(987321)
        X, y = make_classification(n_samples=n_examples, n_classes=3, n_features=8, n_informative=8, n_repeated=0, n_redundant=0)

        self.train_X, self.train_y = X[:n_train, :], y[:n_train]
        self.test_X, self.test_y = X[n_train:, :], y[n_train:]

        np.random.seed(None)

    def _fit_classifier(self):
        self.clf = DecisionTree()

        self.clf.fit(self.train_X, self.train_y)

    def _dfs(self, node):
        if node.left is not None and node.right is not None:
            return self._dfs(node.left) and self._dfs(node.right)
        elif node.left is not None:
            return self._dfs(node.left)
        elif node.right is not None:
            return self._dfs(node.right)
        else:
            return node.class_prediction is not None

    def test_all_leaves_have_class(self):
        self._fit_classifier()
        self.assertTrue(self._dfs(self.clf.root))

    def test_accuracy(self):
        self._fit_classifier()

        y_pred = self.clf.predict(self.test_X)

        acc_clf = accuracy_score(y_pred, self.test_y)

        np.random.seed(987321)
        sk = DecisionTreeClassifier()

        sk.fit(self.train_X, self.train_y)

        y_pred = sk.predict(self.test_X)

        np.random.seed(None)

        acc_sk = accuracy_score(y_pred, self.test_y)

        min_acc, max_acc = min(acc_clf, acc_sk), max(acc_clf, acc_sk)

        percent_difference = (max_acc - min_acc) / max_acc

        # This may seem like a big difference, but it is really random. Some times my implementation is better,
        # sometimes sklearn's implementation is better
        # With a 1000 different random training sets, each with a 1000 random examples, the average difference was less
        # than a single percentage point.
        # This value was chosen so it can be reproducible, as everything is seeded, and we can ensure we are not intro-
        # ducing regressions.

        self.assertLess(percent_difference, 0.07)

    def test_accuracy_regularization(self):
       pass