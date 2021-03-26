import unittest

import numpy as np

from python_ml.Linear.LinearRegression import LinearRegression

class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.reg_normal = LinearRegression()

        np.random.seed(987321)

        self.train_X = 50 * np.random.random(size=(50, 1))
        self.train_y = 3 * self.train_X + 2 + np.random.normal(scale=1.5, size=(50, 1))

        self.reg_normal.fit(self.train_X, self.train_y, mode='normal')

        self.test_X = 50 * np.random.random(size=(10, 1))
        self.test_y = 3 * self.test_X + 2

        np.random.seed(None)

    def test_normal_equation(self):
        y_pred = self.reg_normal.predict(self.test_X)

        mse = np.mean((self.test_y - y_pred) ** 2)

        self.assertTrue(mse < 0.1)

    def test_gd(self):
        y_pred = self.reg_normal.predict(self.test_X)

        mse_normal = np.mean((self.test_y - y_pred) ** 2)

        reg_gd = LinearRegression()

        reg_gd.fit(self.train_X, self.train_y, mode='gd', learning_rate=1e-5)

        y_pred = reg_gd.predict(self.test_X)

        mse_gd = np.mean((self.test_y - y_pred) ** 2)

        min_mse, max_mse = min(mse_normal, mse_gd), max(mse_normal, mse_gd)

        percent_difference = (max_mse - min_mse)/max_mse

        self.assertLess(percent_difference, 0.01)


if __name__ == '__main__':
    unittest.main()