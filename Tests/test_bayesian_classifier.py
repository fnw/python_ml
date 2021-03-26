import unittest
import numpy as np

from python_ml.Bayes.BayesianClassifier import BayesianClassifier

class TestBayesianClassifier(unittest.TestCase):
    def setUp(self):
        self.X1 = np.array([[0,0,0],[0.1,0.1,0.1],[1,1,1],[1.1,1.1,1.1]])
        self.y1 = np.array([0,0,1,1])

        self.expected_mean = np.array([[0.05, 0.05, 0.05], [1.05,1.05,1.05]])
        self.expected_var = 0.0025* np.ones((2,3))

        self.X_test = np.array([[0,0,0],[1,1,1]])
        self.y_test = np.array([0,1])

        self.clf = BayesianClassifier()
        self.clf.fit(self.X1, self.y1)

    def test_fit(self):
        self.assertTrue(np.allclose(self.clf.means, self.expected_mean))
        self.assertTrue(np.allclose(self.clf.vars, self.expected_var))

    def test_predict_likelihood(self):
        probs = self.clf.predict_likelihood(self.X_test)

        true_probs = np.array([[113.33876123530629269539127074620322538667446413625049, 0],[0,113.33876123530629269539127074620322538667446413625049]])

        self.assertTrue(np.allclose(probs,true_probs))

    def test_predict_proba(self):
        probs = self.clf.predict_proba(self.X_test)

        true_probs = np.array([[1, 0],[0,1]])

        self.assertTrue(np.allclose(probs,true_probs))

    def test_predict(self):
        classes = self.clf.predict(self.X_test)

        self.assertTrue(np.array_equal(classes, self.y_test))





if __name__ == '__main__':
    unittest.main()
