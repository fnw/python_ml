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
        self.classifier.fit(self.X,self.y)

        self.assertTrue(np.all(self.single.training_sets[0][0] == self.X))
        self.assertTrue(np.all(self.single.training_sets[0][1] == self.y))


        self.assertEqual(len(self.classifier.training_sets),self.classifier.pool_size,msg='Numero de conjuntos')

        for i in range(self.classifier.pool_size):
            self.assertEqual(self.classifier.training_sets[i][0].shape, self.X.shape,msg='Shape dos X')
            self.assertEqual(self.classifier.training_sets[i][1].shape, self.y.shape,msg='Shape dos y')

        self.assertTrue(self.classifier.has_been_fit,msg='Classifier fit')


