from python_ml.Ensemble.Generation.Bagging import Bagging
from python_ml.Tree.DecisionTree import DecisionTree


class RandomForest(Bagging):
    def __init__(self, pool_size):
        self.has_been_fit = False
        self.pool_size = pool_size
        self.classifiers = []

        for i in range(self.pool_size):
            self.classifiers.append(DecisionTree(use_random_features=True))
