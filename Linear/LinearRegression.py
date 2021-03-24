import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LinearRegression as SKLR
'''
    A simple Gaussian Bayesian Classifier
'''


class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self):
        self._has_been_fit = False
        self.n_examples = None
        self.n_dims = None
        self.weights = None

    def fit(self, X, y, mode='normal', learning_rate=1e-4):
        X, y = check_X_y(X, y)

        self.n_examples, self.n_dims = X.shape

        fit_X = np.ones(shape=(self.n_examples, self.n_dims + 1))

        fit_X[:, 1:] = X

        if mode == 'normal':
            solution = np.dot(np.dot(np.linalg.pinv(np.dot(fit_X.T, fit_X)),
                                     fit_X.T),
                              y)

        elif mode == 'gd':
            solution = np.zeros(shape=self.n_dims+1, dtype=float)
            error = np.ones_like(solution)

            count = 0
            while not np.allclose(error, 0):
                y_pred = np.dot(fit_X, solution)

                error = (y_pred - y)[:, np.newaxis] * fit_X
                error = np.sum(error, axis=0)

                solution -= learning_rate * error
                #print(f'Error it {count}: ', error)
                count += 1

        self.weights = solution
        self._has_been_fit = True

        return self

    def predict(self, X):
        check_is_fitted(self, [])

        X = check_array(X)

        n_examples, n_dims = X.shape

        if n_dims != self.n_dims:
            raise ValueError(f"Invalid number of dimensions. Expected {self.n_dims}, got {n_dims}")

        extended_X = np.ones(shape=(n_examples, n_dims+1), dtype=X.dtype)

        extended_X[:, 1:] = X

        return np.dot(extended_X, self.weights)[:, np.newaxis]






