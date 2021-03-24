import unittest

import numpy as np

from Linear.LinearRegression import LinearRegression


class Test(unittest.TestCase):
    def test_normal_equation(self):
        reg_normal = LinearRegression()

        np.random.seed(987321)

        train_X = 50 * np.random.random(size=(50, 1))
        train_y = 3 * train_X + 2 + np.random.normal(scale=1.5, size=(50, 1))

        reg_normal.fit(train_X, train_y, mode='normal')

        test_X = 50 * np.random.random(size=(10, 1))
        test_y = 3 * test_X + 2

        y_pred = reg_normal.predict(test_X)

        mse = np.mean((test_y - y_pred[:, np.newaxis]) ** 2)

        np.random.seed(None)
        self.assertTrue(mse < 0.1)

    def test_gd(self):
        reg_normal = LinearRegression()

        np.random.seed(987321)

        train_X = 50 * np.random.random(size=(50, 1))
        train_y = 3 * train_X + 2 + np.random.normal(scale=1.5, size=(50, 1))

        reg_normal.fit(train_X, train_y, mode='normal')

        test_X = 50 * np.random.random(size=(10, 1))
        test_y = 3 * test_X + 2

        y_pred = reg_normal.predict(test_X)

        mse_normal = np.mean((test_y - y_pred[:, np.newaxis]) ** 2)

        reg_gd = LinearRegression()

        reg_gd.fit(train_X, train_y, mode='gd', learning_rate=1e-5)

        y_pred = reg_gd.predict(test_X)

        mse_gd = np.mean((test_y - y_pred[:, np.newaxis]) ** 2)

        min_mse, max_mse = min(mse_normal, mse_gd), max(mse_normal, mse_gd)

        percent_difference = (max_mse - min_mse)/max_mse

        self.assertLess(percent_difference, 0.01)




if __name__ == '__main__':
    unittest.main()
