import unittest
import numpy as np

from ..Parzen.Parzen import ParzenDensityEstimator

class TestParzen(unittest.TestCase):
    def setUp(self):
        self.X1 = np.array([[0,0,0],[0.1,0.1,0.1],[1,1,1],[1.1,1.1,1.1]])
        self.y1 = np.array([0,0,1,1])

        self.X_test = np.array([[0,0,0],[1,1,1]])
        self.y_test = np.array([0,1])

        self.clf = ParzenDensityEstimator(h=1)
        self.clf.fit(self.X1, self.y1)

    def test_fit(self):
        self.assertTrue(self.clf._has_been_fit)

    def test_predict_likelihood(self):
        probs = self.clf.predict_likelihood(self.X_test)

    def test_predict_proba(self):
        probs = self.clf.predict_proba(self.X_test)

        self.assertTrue(np.allclose(np.sum(probs, axis=1), np.ones(2)))

    def test_predict(self):
        classes = self.clf.predict(self.X_test)


if __name__ == '__main__':
    unittest.main()