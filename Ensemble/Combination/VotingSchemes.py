import numpy as np
from scipy.stats import mode

def majority_voting(y):
    values, counts = mode(y, axis=-1)
    return values