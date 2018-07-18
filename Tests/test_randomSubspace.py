from unittest import TestCase
import numpy as np

from Ensemble.Generation.RandomSubspace import RandomSubspace

class TestRandomSubspace(TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression()
        self.single = RandomSubspace(lr, pool_size=1, percentage=0.1)
        self.classifier = RandomSubspace(lr, pool_size=5, percentage=0.5)
        self.X, self.y = load_iris(return_X_y=True)

    def test_fit(self):
        self.single.fit(self.X, self.y)
        self.classifier.fit(self.X,self.y)

        self.assertTrue(np.all(self.single.training_sets[0][0] == self.X))
        self.assertTrue(np.all(self.single.training_sets[0][1] == self.y))

        self.assertEqual(len(self.classifier.training_sets),self.classifier.pool_size,msg='Numero de conjuntos')

        for i in range(self.classifier.pool_size):
            self.assertEqual(self.classifier.training_sets[i][0].shape[0], self.X.shape[0],msg='Shape dos X')
            self.assertEqual(self.classifier.training_sets[i][0].shape[1], 2, msg='Num features')
            self.assertEqual(self.classifier.training_sets[i][1].shape, self.y.shape,msg='Shape dos y')

        self.assertTrue(self.classifier.has_been_fit,msg='Classifier fit')


