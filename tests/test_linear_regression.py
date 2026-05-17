import unittest

import numpy as np

from python_ml.Linear.LinearRegression import LinearRegression

class LinearRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(987321)

        self.train_X = 50 * np.random.random(size=(50, 1))
        self.train_y = 3 * self.train_X + 2 + np.random.normal(scale=1.5, size=(50, 1))

        self.test_X = 50 * np.random.random(size=(10, 1))
        self.test_y = 3 * self.test_X + 2

        np.random.seed(None)

    def _fit_classifier(self, regularization=None):
        self.reg_normal = LinearRegression(regularization)
        self.reg_normal.fit(self.train_X, self.train_y, mode='normal')

    def test_normal_equation(self):
        self._fit_classifier()
        y_pred = self.reg_normal.predict(self.test_X)

        mse = np.mean((self.test_y - y_pred) ** 2)

        self.assertTrue(mse < 0.1)

    def test_gd(self):
        self._fit_classifier()
        y_pred = self.reg_normal.predict(self.test_X)

        mse_normal = np.mean((self.test_y - y_pred) ** 2)

        reg_gd = LinearRegression()

        reg_gd.fit(self.train_X, self.train_y, mode='gd', learning_rate=1e-5)

        y_pred = reg_gd.predict(self.test_X)

        mse_gd = np.mean((self.test_y - y_pred) ** 2)

        min_mse, max_mse = min(mse_normal, mse_gd), max(mse_normal, mse_gd)

        percent_difference = (max_mse - min_mse)/max_mse

        self.assertLess(percent_difference, 0.01)

    def test_normal_equation_with_regularization(self):
        self._fit_classifier(regularization=0.1)
        y_pred = self.reg_normal.predict(self.test_X)

        mse = np.mean((self.test_y - y_pred) ** 2)

        self.assertTrue(mse < 0.1)

    def test_gd_with_regularization(self):
        self._fit_classifier(regularization=0.01)
        y_pred = self.reg_normal.predict(self.test_X)

        mse_normal = np.mean((self.test_y - y_pred) ** 2)

        reg_gd = LinearRegression(regularization=0.01)

        reg_gd.fit(self.train_X, self.train_y, mode='gd', learning_rate=1e-5)

        y_pred = reg_gd.predict(self.test_X)

        mse_gd = np.mean((self.test_y - y_pred) ** 2)

        min_mse, max_mse = min(mse_normal, mse_gd), max(mse_normal, mse_gd)

        percent_difference = (max_mse - min_mse)/max_mse

        self.assertLess(percent_difference, 0.02)

    def test_gd_divergence(self):
        reg_gd = LinearRegression(regularization=0.01)

        with self.assertRaises(OverflowError):
            reg_gd.fit(self.train_X, self.train_y, mode='gd', learning_rate=1e-2)


if __name__ == '__main__':
    unittest.main()