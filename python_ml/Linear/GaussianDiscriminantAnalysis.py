import time

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LR

'''
    A simple Gaussian Bayesian Classifier
'''


class GDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self._has_been_fit = False

        self.classes_ = None
        self.class_probabilities = None

        self.X_ = None
        self.y_ = None

        self.n_features = None
        self.n_classes = None

        self.means = None
        self.vars = None
        self.inv_vars = None
        self.normalization_constant = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_, new_y, self.class_probabilities = np.unique(y, return_inverse=True, return_counts=True)
        self.class_probabilities = self.class_probabilities / float(len(y))

        self.n_classes = len(self.classes_)

        self.X_ = X
        self.y_ = new_y

        n_features = X.shape[1]
        n_classes = self.classes_.shape[0]

        self.n_features = n_features
        self.n_classes = n_classes

        self.means = np.zeros((n_classes, n_features))

        vars_accum = 0
        for i in range(n_classes):
            idx = np.where(self.y_ == i)
            idx = idx[0]

            class_elements = X[idx, :]

            self.means[i, :] = np.mean(class_elements, axis=0)

            for elem in class_elements:
                diff_to_mean = elem - self.means[i, :]
                diff_to_mean = diff_to_mean[np.newaxis, :]
                vars_accum += np.dot(diff_to_mean.T, diff_to_mean)

        self.vars = vars_accum / len(y)
        self.inv_vars = np.linalg.pinv(self.vars)

        self.normalization_constant = 1.0 / ((np.linalg.det(self.vars) ** 0.5) *
                                             ((2 * np.pi) ** (self.n_features / 2)))


        return self

    def predict_likelihood(self, X):
        n_examples = X.shape[0]

        likelihood = np.zeros((n_examples, self.n_classes))

        for i in range(n_examples):
            for j in range(self.n_classes):
                diff_to_mean = X[i, :] - self.means[j, :]

                l = np.exp(-0.5 * np.dot(np.dot(diff_to_mean, self.inv_vars),
                                         diff_to_mean.T)
                           )

                l *= self.normalization_constant

                likelihood[i, j] = l

        return likelihood

    def predict_proba(self, X):
        n_examples = X.shape[0]

        probas = self.predict_likelihood(X)

        probas *= self.class_probabilities
        probas /= np.sum(probas, axis=1).reshape(n_examples, 1)

        probas[np.isnan(probas)] = 0

        return probas

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X)

        probas = self.predict_proba(X)

        return self.classes_[np.argmax(probas, axis=1)]
