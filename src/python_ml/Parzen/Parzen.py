import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

'''
    A Parzen Density Estimator, using an Epanechnikov kernel.
    
    Parameters:
    h: int (Default = 1):
        Size of the window for the density estimation.
'''

class ParzenDensityEstimator(BaseEstimator,ClassifierMixin):
    def __init__(self, h=1):
        self.h = float(h)
        self._has_been_fit = False

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_, new_y, self.class_probabilities = np.unique(y, return_inverse=True, return_counts=True)
        self.class_probabilities = self.class_probabilities/float(len(y))

        self.n_classes = len(self.classes_)

        self.X_ = X
        self.y_ = new_y

        self._idx_by_classes = {}

        for c in range(self.n_classes):
            self._idx_by_classes[c] = np.where(self.y_ == c)[0]


        self._has_been_fit = True

        return self

    def predict_likelihood(self,X):
        n_examples = X.shape[0]
        n_features = X.shape[1]

        likelihood = np.zeros((n_examples, self.n_classes))

        sqrt5 = np.sqrt(5)

        constant = 3.0/(4.0*sqrt5)

        for i in range(n_examples):
            for j in range(self.n_classes):
                l = (X[i,:] - self.X_[self._idx_by_classes[j],:])/self.h

                idx = np.where(np.abs(l) >= sqrt5)

                l *= l
                l /= 5
                l = 1 - l
                l *= constant

                l[idx] = 0

                l = np.prod(l, axis=1)
                l = np.sum(l)

                likelihood[i,j] = l

        #We use logarithms to avoid overflow issues caused by exponentiation.
        likelihood *= np.exp(-(np.log(n_examples) + n_features*np.log(self.h)))

        return likelihood


    def predict_proba(self,X):
        n_examples = X.shape[0]

        probas = self.predict_likelihood(X)

        probas *= self.class_probabilities

        probas /= np.sum(probas,axis=1).reshape(n_examples,1)

        probas[np.isnan(probas)] = 0

        return probas

    def predict(self,X):
        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X)

        probas = self.predict_proba(X)

        return self.classes_[np.argmax(probas, axis=1)]

