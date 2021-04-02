import unittest

import numpy as np

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from python_ml.Ensemble.Generation.RandomForest import RandomForest


class TestDecisionTrees(unittest.TestCase):
    def setUp(self) -> None:
        n_examples = 100
        n_train = int(0.75 * n_examples)

        np.random.seed(987321)
        X, y = make_classification(n_samples=n_examples, n_classes=3, n_features=8, n_informative=8, n_repeated=0, n_redundant=0)

        self.train_X, self.train_y = X[:n_train, :], y[:n_train]
        self.test_X, self.test_y = X[n_train:, :], y[n_train:]

        np.random.seed(None)

    def _fit_classifier(self):
        np.random.seed(987321)
        self.clf = RandomForest(20)

        self.clf.fit(self.train_X, self.train_y)
        np.random.seed(None)

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
        self.assertTrue(all((self._dfs(tree.root) for tree in self.clf.classifiers)))

    def test_perfect_accuracy(self):
        # This example should be perfectly separable, and have an accuracy of 100%
        # This simulates college admissions given a test score and whether the person has extracurriculars.
        # I came up with it, and the semantics don't matter, what matters is that the class label is defined by a simple
        # rule: score > 70 and has extracurriculars.

        n_examples = 1000

        X = []

        np.random.seed(987321)

        for i in range(n_examples):
            grade = np.random.randint(100)
            has_extracurricular = np.random.choice(["no", "yes"])

            X.append([grade, has_extracurricular])

        X = np.array(X, dtype=object)

        y = ((X[:, 0] > 70) & (X[:, 1] == "yes")).astype(int)

        clf = RandomForest(20)

        clf.fit(X, y)

        test_X = []
        for i in range(n_examples):
            grade = np.random.randint(100)
            has_extracurricular = np.random.choice(["no", "yes"])

            test_X.append([grade, has_extracurricular])

        np.random.seed(None)

        test_X = np.array(test_X, dtype=object)

        test_y = ((test_X[:, 0] > 70) & (test_X[:, 1] == "yes")).astype(int)

        y_pred = clf.predict(test_X)

        self.assertEqual(accuracy_score(y_pred, test_y), 1.0)

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

        # The random forest should be better than a single classifier.
        self.assertGreaterEqual(acc_clf, acc_sk)
