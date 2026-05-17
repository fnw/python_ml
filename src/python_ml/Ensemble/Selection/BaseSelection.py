from copy import deepcopy

import numpy as np

from sklearn.neighbors import NearestNeighbors

class BaseSelection(object):
    def __init__(self, X_val, y_val, n_neighbors):
        self.__X = X_val
        self.__y = y_val
        self.k = n_neighbors

        self.__knn = NearestNeighbors(n_neighbors=self.k)

        self.__knn.fit(X_val, y_val)

    def kNearest(self, X_query):
        _, ind = self.__knn.kneighbors(X=X_query, n_neighbors=self.k)
        return ind

    def predict_nearest(self, X_query, ensemble):
        ind = self.kNearest(X_query)

        actual_k = self.k
        num_queries = X_query.shape[0]

        ind = ind.flatten()

        X_val_nearest = self.__X[ind,:]
        y_val_nearest = self.__y[ind]

        predictions = ensemble.predict(X_val_nearest)

        predictions = np.reshape(predictions, (num_queries, actual_k, predictions.shape[1]))
        y_val_nearest = np.reshape(y_val_nearest, (num_queries, actual_k))

        return predictions, y_val_nearest

    def set_n_neighbors(self, k):
        self.k = k

    def select(self, X_query, ensemble):
        pass