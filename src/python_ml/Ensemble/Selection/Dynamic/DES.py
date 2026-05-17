from Ensemble.Selection.BaseSelection import BaseSelection
import numpy as np
from Ensemble.Combination.VotingSchemes import majority_voting

class KNORAU(BaseSelection):
    def __init__(self, X_val, y_val, n_neighbors):
        super(KNORAU, self).__init__(X_val, y_val, n_neighbors)

    def best_local_accuracy(self, predictions, actual_labels):
        num_queries = actual_labels.shape[0]
        n_neighbors = self.k

        n_classifiers = predictions.shape[2]

        should_use = np.zeros((num_queries, n_classifiers),dtype=bool)
        weights = np.zeros((num_queries, n_classifiers),dtype=np.float32)

        for i in range(num_queries):
            pred_neighbors = predictions[i,:,:]
            labels_neighbors = actual_labels[i,:]

            labels_neighbors = np.reshape(labels_neighbors,(n_neighbors, 1))
            labels_neighbors = np.tile(labels_neighbors,(1,n_classifiers))

            correct = (pred_neighbors == labels_neighbors)

            weights[i,:] = np.sum(correct, axis=0)
            any_correct = np.any(correct, axis=0)

            should_use[i,:] = any_correct

        return should_use, weights

    def select(self, X_query, ensemble):
        predictions, actual_labels = self.predict_nearest(X_query, ensemble)
        should_use, weights = self.best_local_accuracy(predictions, actual_labels)

        return should_use, weights

    def predict_subset(self, X_query, ensemble, should_use, weights):
        predictions = []

        n_classifiers = len(ensemble.classifiers)

        for i in range(n_classifiers):
            if should_use[i]:
                prediction = ensemble.classifiers[i].predict(X_query)

                for j in range(weights[i]):
                    predictions.append(prediction)

        predictions = np.array(predictions)
        predictions = predictions.reshape((1,n_classifiers))

        return majority_voting(predictions)


    def select_and_predict(self, X_query, ensemble):
        num_queries = X_query.shape[0]
        should_use, weights = self.select(X_query, ensemble)

        predictions = np.zeros(num_queries)

        for i in range(num_queries):
           predictions[i] = self.predict_subset(X_query[i,:], ensemble, should_use[i,:], weights[i,:])

        return predictions