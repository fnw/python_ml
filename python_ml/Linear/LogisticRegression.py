import numpy as np

import time

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LR
'''
    A simple Gaussian Bayesian Classifier
'''


def logistic_function(x):
    return 1/(1 + np.exp(-x))


class LogisticRegression(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=1e-4, max_iter=250000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self._has_been_fit = False
        self.n_examples = None
        self.n_dims = None
        self.weights = None

    def fit(self, X, y, print_error=False):
        X, y = check_X_y(X, y)

        self.n_examples, self.n_dims = X.shape

        fit_X = np.ones(shape=(self.n_examples, self.n_dims + 1))

        fit_X[:, 1:] = X

        solution = np.zeros(shape=self.n_dims+1, dtype=float)
        error = np.ones_like(solution)

        iterations = 0
        while not np.allclose(error, 0) and iterations < self.max_iter:
            logits = np.dot(fit_X, solution)
            probas = logistic_function(logits)

            # theta_j = theta_j - sum(yi - y_pred)*xj
            error = (y - probas)[:, np.newaxis] * fit_X
            error = np.sum(error, axis=0)

            # Maximizing the likelihood.
            solution += self.learning_rate * error

            if print_error:
                print(f'Error it {iterations}: ', error)

            iterations += 1

        self.weights = solution
        self._has_been_fit = True

        return self

    def predict_proba(self, X):
        check_is_fitted(self, [])

        X = check_array(X)

        n_examples, n_dims = X.shape

        if n_dims != self.n_dims:
            raise ValueError(f"Invalid number of dimensions. Expected {self.n_dims}, got {n_dims}")

        extended_X = np.ones(shape=(n_examples, n_dims + 1), dtype=X.dtype)

        extended_X[:, 1:] = X

        return logistic_function(np.dot(extended_X, self.weights)[:, np.newaxis])

    def predict(self, X):
        probas = self.predict_proba(X)

        y_pred = np.zeros_like(probas).astype(int)

        y_pred[probas > 0.5] = 1

        return y_pred
