import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class KMeans(TransformerMixin, ClusterMixin, BaseEstimator):

    def __init__(
        self,
        n_clusters,
        max_iter=300,
        tol=1e-4,
    ):

        self.n_clusters = n_clusters
        self.centroids = None
        self.cluster_assignments = None
        self.cluster_labels = None
        self.labels = None

        self.max_iter = max_iter
        self.tol = tol

        self.has_been_fit = False

    def fit(self, X, y=None):
        n_samples = X.shape[0]
        random_idxs = np.random.randint(0, n_samples, size=self.n_clusters)

        self.centroids = X[random_idxs]
        self.cluster_labels = np.zeros(self.n_clusters)
        self.cluster_assignments = np.zeros(n_samples)
        self.labels = np.zeros(n_samples)

        centroid_diffs = 1e6
        it = 0

        while centroid_diffs > self.tol and it < self.max_iter:
            centroid_diffs = 0

            dist_to_centroids = np.sum((X[:, np.newaxis, :] - self.centroids) ** 2, axis=-1)
            self.cluster_assignments = np.argmin(dist_to_centroids, axis=1)

            for i in range(self.n_clusters):
                samples_this_clusters = X[self.cluster_assignments == i, :]
                new_centroid = np.mean(samples_this_clusters, axis=0)

                centroid_diffs += np.sum((new_centroid - self.centroids[i, :]) ** 2, axis=-1)

                self.centroids[i, :] = new_centroid

            centroid_diffs /= self.n_clusters
            it += 1

        dist_to_centroids = np.sum((X[:, np.newaxis, :] - self.centroids) ** 2, axis=-1)
        self.cluster_assignments = np.argmin(dist_to_centroids, axis=1)

        # This could have been a one-liner, but this is clearer
        for i in range(self.n_clusters):
            idx_this_cluster = self.cluster_assignments == i
            classes_this_cluster = y[idx_this_cluster]
            most_common_class = mode(classes_this_cluster).mode[0]
            self.cluster_labels[i] = most_common_class
            self.labels[idx_this_cluster] = most_common_class

        self.has_been_fit = True
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels

    def predict(self, X, sample_weight=None):
        if not self.has_been_fit:
            raise ValueError('KMeans object has not been fit')

        dist_to_centroids = np.sum((X[:, np.newaxis, :] - self.centroids) ** 2, axis=-1)
        cluster_assignments = np.argmin(dist_to_centroids, axis=1)

        return self.cluster_labels[cluster_assignments]
