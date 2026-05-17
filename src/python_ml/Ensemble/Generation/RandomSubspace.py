import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from python_ml.Ensemble.Combination.VotingSchemes import majority_voting

class RandomSubspace(object):
    def __init__(self, base_classifier, pool_size, percentage=0.5):
        self.has_been_fit = False
        self.pool_size = pool_size
        self.percentage = percentage
        self.classifiers = []

        for i in range(self.pool_size):
            self.classifiers.append(clone(base_classifier))

        self.training_sets = []
        self.selected_features = []

    def _generate_set(self, X, y):
        num_features_original = X.shape[1]

        if self.pool_size == 1:
            X_set = X
            y_set = y
            self.selected_features.append(np.arange(num_features_original))
        else:
            idx = np.random.choice(num_features_original, int(np.round(self.percentage * num_features_original)), replace=False)
            X_set = X[:, idx]
            y_set = y[:]
            self.selected_features.append(idx)

        return X_set, y_set

    def fit(self, X, y):
        if not self.has_been_fit:
            for i, clf in enumerate(self.classifiers):
                X_this, y_this = self._generate_set(X, y)
                clf.fit(X_this, y_this)

        self.has_been_fit = True

    def predict(self, X, voting_scheme='majority vote'):
        if not self.has_been_fit:
            raise Exception('Classifier has not been fit')

        if isinstance(voting_scheme,str):
            if voting_scheme == 'majority vote':
                voting_scheme = majority_voting

        predictions = []

        for i, clf in enumerate(self.classifiers):
            selected_features = self.selected_features[i]
            predictions.append(clf.predict(X[:,selected_features]))

        predictions = np.array(predictions).T

        return voting_scheme(predictions)

    def score(self,X,y):
        return accuracy_score(y, self.predict(X))



