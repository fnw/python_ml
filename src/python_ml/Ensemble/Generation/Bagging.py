import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from python_ml.Ensemble.Combination.VotingSchemes import majority_voting


def all_classes_present(y_prime, y):
    return set(y) == set(y_prime)


def add_missing_classes(X_prime, X, y_prime, y, minimum=1, seed=None):
    missing_classes = set(y) - set(y_prime)

    for c in missing_classes:
        idx = np.where(y == c)[0]

        if seed is not None:
            np.random.seed(seed)

        idx_selected = np.random.choice(idx, size=minimum)

        X_prime = np.vstack((X_prime, X[idx_selected, :]))
        y_prime = np.concatenate((y_prime, np.array(y[idx_selected])))

    return X_prime, y_prime


class Bagging(object):
    def __init__(self, base_classifier, pool_size):
        self.has_been_fit = False
        self.pool_size = pool_size
        self.classifiers = []

        for i in range(self.pool_size):
            self.classifiers.append(clone(base_classifier))

        self.training_sets = []

    def _generate_sets(self, X, y):
        size_original = len(y)

        if self.pool_size == 1:
            X_set = X
            y_set = y
        else:
            idx = np.random.choice(size_original, size_original)
            X_set = X[idx, :]
            y_set = y[idx]

            # Depending on the elements selected for the bootstrap, some classes may not be present in the bootstrap
            if not all_classes_present(y_set, y):
                X_set, y_set = add_missing_classes(X_set, X, y_set, y)

        return X_set, y_set

    def fit(self, X, y):
        if not self.has_been_fit:
            for i, clf in enumerate(self.classifiers):
                X_this, y_this = self._generate_sets(X, y)
                clf.fit(X_this, y_this)

            self.has_been_fit = True

    def predict(self, X, voting_scheme='majority vote'):
        if not self.has_been_fit:
            raise Exception('Classifier has not been fit')

        if isinstance(voting_scheme,str):
            if voting_scheme == 'majority vote':
                voting_scheme = majority_voting

        predictions = []

        for clf in self.classifiers:
            predictions.append(clf.predict(X))

        predictions = np.array(predictions).T

        return voting_scheme(predictions)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))



