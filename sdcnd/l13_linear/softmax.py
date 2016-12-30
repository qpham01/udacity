# Solution is available in the other "solution.py" tab
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    """
    if isinstance(x, list):
        x = np.array(x)
    if len(x.shape) == 1 or x.shape[1] == 1:
        ex = np.exp(x)
        return ex / ex.sum()
    result = np.zeros(x.shape)
    for col in range(x.shape[1]):
        ex = np.exp(x[:, col])
        result[:, col] = ex / ex.sum()
    return result
    """

logits = np.array([3.0, 1.0, 0.2])
logits = np.array([
    [1, 2, 3, 6],
    [2, 4, 5, 6],
    [3, 8, 7, 6]])
print(softmax(logits))
