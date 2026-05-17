from enum import Enum
import numpy as np

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class VariableType(Enum):
    CATEGORICAL = 1
    NUMERIC = 2


class TreeNode:
    def __init__(self, split_variable_idx, split_variable_value, split_variable_type, class_prediction=None):
        self.split_variable_idx = split_variable_idx
        self.split_variable_value = split_variable_value
        self.split_variable_type = split_variable_type

        self.left = None
        self.right = None
        self.class_prediction = class_prediction


class DecisionTree:
    def __init__(self, algorithm='CART', use_random_features=False):
        self.algorithm = algorithm
        self.root = None
        self.variable_types = []

        self.class_counts = None
        self.class_priors = None
        self.n_classes = None

        self.use_random_features = use_random_features
        self.n_random_features = None

    def _calc_joint_class_node(self, y, class_idx):
        n_j = self.class_counts[class_idx]
        eta_j = self.class_priors[class_idx]

        n_j_t = np.sum(y == class_idx)

        return (eta_j * n_j_t)/n_j

    def _calc_conditional_class_node(self, joints, class_idx):
        return joints[class_idx] / np.sum(joints)

    def _calculate_impurity_and_probabilities(self, y):
        joints = np.array([self._calc_joint_class_node(y, class_idx) for class_idx in range(self.n_classes)])
        conditionals = np.array([self._calc_conditional_class_node(joints, class_idx) for class_idx in range(self.n_classes)])

        # Pairwise products of class conditionals
        products_matrix = np.dot(conditionals[:, np.newaxis], conditionals[np.newaxis, :])

        # Misclassification cost (class, class) is equal to zero
        products_matrix = products_matrix * (1 - np.eye(*products_matrix.shape))

        return np.sum(products_matrix), joints, conditionals

    def _calculate_information_gain(self, X, y, already_filtered_idx, variable_idx):
        impurity, joints, conditionals = self._calculate_impurity_and_probabilities(y[already_filtered_idx])
        p_t = np.sum(joints)

        variable_values = X[:, variable_idx]
        variable_values = variable_values[already_filtered_idx]
        variable_values = np.unique(variable_values)

        max_gain = None
        best_value = None

        for value in variable_values:
            if self.variable_types[variable_idx] == VariableType.CATEGORICAL:
                idx_this_split = X[:, variable_idx] == value
            elif self.variable_types[variable_idx] == VariableType.NUMERIC:
                idx_this_split = X[:, variable_idx] <= value

            idx_left = already_filtered_idx & idx_this_split
            idx_right = already_filtered_idx & np.logical_not(idx_this_split)

            i_l, joints_l, conditionals_l = self._calculate_impurity_and_probabilities(y[idx_left])
            i_r, joints_r, conditionals_r = self._calculate_impurity_and_probabilities(y[idx_right])

            p_l = np.sum(joints_l)
            p_r = np.sum(joints_r)

            gain = impurity - (p_l/p_t)*i_l - (p_r/p_t)*i_r
            # print(f'Value {value}, Gain: {gain}')

            if max_gain is None or gain > max_gain:
                max_gain = gain
                best_value = value

        return max_gain, best_value

    def _build_tree(self, X, y, splits=[]):
        X_copy = X.copy()
        num_variables = X.shape[1]

        already_used_variables = set()

        # This will be used to subset the elements of the training set according to where they are positioned in
        # each split.
        idx = np.ones(len(y), dtype=bool)

        # Subset the variables considering each split until the current depth
        # left indicates whether we are the left or right subtree
        for split_variable, split_value, left in splits:
            already_used_variables.add(split_variable)

            split_type = self.variable_types[split_variable]

            if split_type == VariableType.CATEGORICAL:
                if left:
                    idx = idx & (X[:, split_variable] == split_value)
                else:
                    idx = idx & (X[:, split_variable] != split_value)
            elif split_type == VariableType.NUMERIC:
                if left:
                    idx = idx & (X[:, split_variable] <= split_value)
                else:
                    idx = idx & (X[:, split_variable] > split_value)

        y_filtered = y[idx]

        classes_present, counts = np.unique(y_filtered, return_counts=True)

        # If there's only one class at this node, then it's a terminal node and we use it to make a prediction.
        if len(classes_present) == 1:
            return TreeNode(None, None, None, class_prediction=classes_present[0])

        variables_remaining = [i for i in range(num_variables) if i not in already_used_variables]

        # If we can't split anymore, just return the most frequent class
        if not variables_remaining:
            max_class_idx = np.argmax(counts)
            max_class = classes_present[max_class_idx]

            return TreeNode(None, None, None, class_prediction=max_class)

        # Use random features, for Random Forests
        if self.use_random_features:
            if self.n_random_features < len(variables_remaining):
                variables_remaining = np.random.choice(variables_remaining, self.n_random_features, replace=False)

        information_gains, split_values = zip(*(self._calculate_information_gain(X, y, idx, variable_idx)
                                                for variable_idx in variables_remaining))

        idx_max_gain = np.argmax(information_gains)

        new_split_variable = variables_remaining[idx_max_gain]
        new_split_value = split_values[idx_max_gain]

        new_node = TreeNode(new_split_variable, new_split_value, self.variable_types[new_split_variable])

        # Create the children of the node
        new_node.left = self._build_tree(X, y, splits + [(new_split_variable, new_split_value, True)])
        new_node.right = self._build_tree(X, y, splits + [(new_split_variable, new_split_value, False)])

        return new_node

    def fit(self, X, y):
        unique_classes, counts = np.unique(y, return_counts=True)
        self.class_counts = counts
        self.class_priors = counts / np.sum(counts)

        self.n_classes = len(self.class_counts)

        n_dims = X.shape[1]

        if self.use_random_features:
            self.n_random_features = int(np.log2(n_dims) + 1)

        for d in range(n_dims):
            value = X[0, d]

            if isinstance(value, str) or isinstance(value, bool):
                self.variable_types.append(VariableType.CATEGORICAL)
            else:
                self.variable_types.append(VariableType.NUMERIC)

        self.root = self._build_tree(X, y)

    def _walk_tree(self, node: TreeNode, X):
        if node is None:
            return None
        if node.class_prediction is not None:
            return node.class_prediction
        else:
            split_variable_idx = node.split_variable_idx
            split_variable_value = node.split_variable_value
            split_variable_type = node.split_variable_type

            if split_variable_type == VariableType.CATEGORICAL:
                left = X[split_variable_idx] == split_variable_value
            elif split_variable_type == VariableType.NUMERIC:
                left = X[split_variable_idx] <= split_variable_value

            first = node.left if left else node.right
            retry = node.right if left else node.left

            ret = self._walk_tree(first, X)

            if ret is None:
                print('Oi')
                ret = self._walk_tree(retry, X)

            return ret

    def predict(self, X):
        predictions = []

        for elem in X:
            predictions.append(self._walk_tree(self.root, elem))

        predictions = [pred if pred is not None else np.random.randint(self.n_classes) for pred in predictions]

        return np.array(predictions)
