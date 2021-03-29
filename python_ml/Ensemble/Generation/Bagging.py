import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from python_ml.Ensemble.Combination.VotingSchemes import majority_voting


class Bagging(object):
    def __init__(self, base_classifier, pool_size):
        self.has_been_fit = False
        self.pool_size = pool_size
        self.classifiers = []

        for i in range(self.pool_size):
            self.classifiers.append(clone(base_classifier))

        self.training_sets = []

    def __generate_sets(self,X,y):
        size_original = len(y)

        if self.pool_size == 1:
            X_set = X
            y_set = y
            self.training_sets.append((X_set, y_set))
        else:
            for i in range(self.pool_size):
                idx = np.random.choice(size_original, size_original)
                X_set = X[idx, :]
                y_set = y[idx]
                self.training_sets.append((X_set, y_set))

    def fit(self, X, y):
        if not self.has_been_fit:
            self.__generate_sets(X,y)
            for i, clf in enumerate(self.classifiers):
                X_this, y_this = self.training_sets[i][0], self.training_sets[i][1]
                clf.fit(X_this,y_this)

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



