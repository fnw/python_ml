from unittest import TestCase

import numpy as np
from python_ml.Ensemble.Generation.Bagging import Bagging

class TestBagging(TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression()
        self.single = Bagging(lr, pool_size=1)
        self.classifier = Bagging(lr, pool_size=5)
        self.X, self.y = load_iris(return_X_y=True)

    def test_fit(self):
        self.single.fit(self.X, self.y)
        self.classifier.fit(self.X, self.y)

        self.assertTrue(self.classifier.has_been_fit, msg='Classifier fit')

    def test_predict(self):
        self.classifier.fit(self.X, self.y)

        y_pred = self.classifier.predict(self.X)

        self.assertEqual(len(y_pred), len(self.X))


