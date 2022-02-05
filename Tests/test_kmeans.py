import unittest

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LR

from python_ml.Clustering.kmeans import KMeans


class TestKMeans(unittest.TestCase):
    def setUp(self):
        n_centers = 4
        self.X, self.y, centers = make_blobs(n_samples=500, n_features=2, centers=n_centers, return_centers=True, random_state=42)
        self.km = KMeans(n_clusters=n_centers)
        self.orig_predictions = self.km.fit_predict(self.X, self.y)

    def test_predictions(self):
        self.assertTrue(np.all(self.y == self.orig_predictions))

    def test_predict(self):
        self.assertTrue(np.all(self.km.predict(self.X) == self.orig_predictions))


if __name__ == '__main__':
    unittest.main()
