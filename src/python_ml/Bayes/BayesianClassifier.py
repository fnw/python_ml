import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

'''
    A simple Gaussian Bayesian Classifier
'''

class BayesianClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self._has_been_fit = False

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
        self.vars = np.zeros((n_classes, n_features))

        for i in range(n_classes):
            idx = np.where(self.y_ == i)
            idx = idx[0]

            self.means[i,:] = np.mean(X[idx,:], axis=0)
            self.vars[i,:] = np.var(X[idx,:],axis=0)

        self.log_vars = np.log(self.vars)

        return self

    def predict_likelihood(self,X):
        n_examples = X.shape[0]

        likelihood = np.zeros((n_examples, self.n_classes))

        constant = -0.5*np.log(2*np.pi)*self.n_features

        for i in range(n_examples):
            l = np.square(self.means - X[i, :])
            l = l / self.vars
            l = -0.5 * np.sum(l, axis=1)

            l += -0.5 * np.sum(self.log_vars, axis=1)
            l += constant

            likelihood[i, :] = l

        likelihood = np.exp(likelihood)

        return likelihood

    def predict_proba(self,X):
        n_examples = X.shape[0]

        probas = self.predict_likelihood(X)

        probas *= self.class_probabilities
        probas /= np.sum(probas, axis=1).reshape(n_examples, 1)

        probas[np.isnan(probas)] = 0

        return probas

    def predict(self,X):
        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X)

        probas = self.predict_proba(X)

        return self.classes_[np.argmax(probas, axis=1)]



