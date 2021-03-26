from python_ml.Ensemble.Selection.BaseSelection import BaseSelection
import numpy as np

class OLA(BaseSelection):
    def __init__(self, X_val, y_val, n_neighbors):
        super(OLA, self).__init__(X_val, y_val, n_neighbors)

    def best_local_accuracy(self, predictions, actual_labels):
        num_queries = actual_labels.shape[0]
        n_neighbors = self.k

        n_classifiers = predictions.shape[2]

        best_classifiers = np.zeros((num_queries),dtype=int)

        for i in range(num_queries):
            pred_neighbors = predictions[i,:,:]
            labels_neighbors = actual_labels[i,:]

            labels_neighbors = np.reshape(labels_neighbors,(n_neighbors, 1))
            labels_neighbors = np.tile(labels_neighbors,(1,n_classifiers))

            correct = (pred_neighbors == labels_neighbors)
            correct = np.sum(correct, axis=0)

            best = np.argmax(correct)
            best_classifiers[i] = best

        return best_classifiers

    def select(self, X_query, ensemble):
        predictions, actual_labels = self.predict_nearest(X_query, ensemble)
        best = self.best_local_accuracy(predictions, actual_labels)

        return best

    def select_and_predict(self, X_query, ensemble):
        num_queries = X_query.shape[0]
        best_classifiers = self.select(X_query, ensemble)

        predictions = np.zeros(num_queries)

        for i in range(num_queries):
            predictions[i] = ensemble.classifiers[best_classifiers[i]].predict(X_query[i,:].reshape(1,-1))

        return predictions

class LCA(BaseSelection):
    def __init__(self, X_val, y_val, n_neighbors):
        super(LCA, self).__init__(X_val, y_val, n_neighbors)

    def best_local_class_accuracy(self, predictions_nearest, predictions_query, actual_labels):
        num_queries = actual_labels.shape[0]
        n_neighbors = self.k

        n_classifiers = predictions_nearest.shape[2]

        best_classifiers = np.zeros((num_queries),dtype=int)

        for i in range(num_queries):
            pred_neighbors = predictions_nearest[i,:,:]
            labels_neighbors = actual_labels[i,:]
            label_query = predictions_query[i]

            labels_neighbors = np.reshape(labels_neighbors,(n_neighbors, 1))
            labels_neighbors = np.tile(labels_neighbors,(1,n_classifiers))

            same_predicted_class = pred_neighbors == label_query

            correct = np.logical_and((pred_neighbors == labels_neighbors),same_predicted_class)
            correct = np.sum(correct, axis=0)
            correct = correct/np.sum(same_predicted_class, axis=0).astype(float)

            is_division_by_zero = np.logical_not(np.isfinite(correct))

            correct[is_division_by_zero] = 0

            best = np.argmax(correct)
            best_classifiers[i] = best

        return best_classifiers

    def select(self, X_query, ensemble):
        predictions_nearest, actual_labels = self.predict_nearest(X_query, ensemble)
        predictions_query = ensemble.predict(X_query)
        best = self.best_local_class_accuracy(predictions_nearest, predictions_query, actual_labels)

        return best

    def select_and_predict(self, X_query, ensemble):
        num_queries = X_query.shape[0]
        best_classifiers = self.select(X_query, ensemble)

        predictions = np.zeros(num_queries)

        for i in range(num_queries):
            predictions[i] = ensemble.classifiers[best_classifiers[i]].predict(X_query[i,:].reshape(1,-1))

        return predictions






