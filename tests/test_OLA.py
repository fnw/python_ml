import unittest

import numpy as np

from python_ml.Ensemble.Selection.BaseSelection import BaseSelection
from python_ml.Ensemble.Selection.Dynamic.DCS import OLA
from python_ml.Ensemble.Generation.Bagging import Bagging

class test_ola(unittest.TestCase):
    def setUp(self):
        self.n_queries = 2
        self.n_neighbors = 3
        self.n_classifiers = 5

        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression()
        self.bag = Bagging(lr, pool_size=self.n_classifiers)
        self.X, self.y = load_iris(return_X_y=True)

        np.random.seed(5)
        self.predictions = np.random.randint(low=np.min(self.y), high=np.max(self.y), size=(self.n_queries, self.n_neighbors, self.n_classifiers))

        self.bag.fit(self.X, self.y)

        self.selector = OLA(self.X, self.y, n_neighbors=self.n_neighbors)

    def test_local_accuracy(self):
        _, actual_labels = self.selector.predict_nearest(self.X[:self.n_queries,:], self.bag)

        a = self.selector.best_local_accuracy(self.predictions, actual_labels)

        self.assertEqual(a.shape[0], self.n_queries)

    def test_select_and_predict(self):
        predictions = self.selector.select_and_predict(self.X[:3], self.bag)

        self.assertEqual(predictions.shape, (3,))

if __name__ == '__main__':
    unittest.main()
