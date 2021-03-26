import numpy as np
from scipy.stats import mode


def majority_voting(y):
    y = np.array(y)

    if len(y.shape) < 2:
        values, counts = mode(y)
    else:
        values, counts = mode(y, axis=1)

    return values