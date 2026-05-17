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


def softmax(x):
    exp = np.exp(x)

    return exp/(np.sum(exp, axis=1)[:, np.newaxis])


class SoftmaxRegressionClassifier(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=1e-4, max_iter=250000, regularization=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self._has_been_fit = False
        self.n_examples = None
        self.n_dims = None
        self.n_classes = None
        self.weights = None

        self.regularization = regularization

    def fit(self, X, y, print_error=False):
        #X, y = check_X_y(X, y)

        self.n_examples, self.n_dims = X.shape
        self.n_classes = y.shape[1]

        fit_X = np.ones(shape=(self.n_examples, self.n_dims + 1))

        fit_X[:, 1:] = X

        solution = np.zeros(shape=(self.n_dims+1, self.n_classes), dtype=float)
        error = np.ones_like(solution)

        iterations = 0
        while not np.allclose(error, 0) and iterations < self.max_iter:
            logits = np.dot(fit_X, solution)
            probas = softmax(logits)

            # regularization_term = self.regularization * solution if self.regularization is not None else 0

            # probas (n_examples, n_classes), y (n_examples, n_classes), x (n_examples, n_dims), weights (n_dims, n_classes)

            for c in range(self.n_classes):
                temp_matrix = y * probas[:, c].reshape(self.n_examples, 1)
                R = np.dot(fit_X.T, temp_matrix)

                temp_matrix2 = fit_X * y[:, c].reshape(self.n_examples, 1)

                error = (1.0/self.n_examples) * (R.sum(axis=1) - np.sum(temp_matrix2, axis=0))
                #print(f"Error on it {iterations}: {error[:,0]}")

                if np.any(np.isnan(error)):
                    raise OverflowError(
                        "Optimization has diverged. Try again with lower learning rate or stronger regularization.")

                # We are minimizing the NLL, because I started the derivation like this and it's something like nine
                # pages long.
                solution[:, c] -= self.learning_rate * error

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

        return softmax(np.dot(extended_X, self.weights))

    def predict(self, X):
        probas = self.predict_proba(X)

        y_pred = np.argmax(probas, axis=1)

        return y_pred

def make_one_hot(y):
    n = len(y)
    max = np.max(y)

    ret = np.zeros(shape=(n, max + 1), dtype=float)

    ret[np.arange(n), y] = 1.0

    return ret