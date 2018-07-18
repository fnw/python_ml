import unittest

import numpy as np

from Ensemble.Selection.BaseSelection import BaseSelection
from Ensemble.Generation.Bagging import Bagging

class TestBaseSelection(unittest.TestCase):
    def setUp(self):
        self.true_labels = np.array([1, 1, 0, 0, 1, 0])

        self.predictions1 = np.array([1, 0, 0, 0, 0, 0])
        self.predictions2 = np.array([0, 1, 0, 0, 1, 1])
        self.predictions3 = np.array([1, 0, 0, 0, 0, 1])

        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression()
        self.bag = Bagging(lr, pool_size=5)
        self.X, self.y = load_iris(return_X_y=True)

        self.bag.fit(self.X, self.y)

        self.selector = BaseSelection(self.X, self.y, n_neighbors=3)

    def test_predict_nearest(self):
        predictions, actual_values = self.selector.predict_nearest(self.X[0:2,:], self.bag)

        self.assertEqual(predictions.shape, (2,3,1))
        self.assertEqual(actual_values.shape, (2,3))