"""Softmax."""

import numpy as np

scores = np.array([3.0, 1.0, 0.2])


"""
given a list or one-dimensional array (which is interpreted as a column vector representing a single sample), like:
scores = [1.0, 2.0, 3.0]
It should return a one-dimensional array of the same length, i.e. 3 elements:
print softmax(scores)
[ 0.09003057  0.24472847  0.66524096]

Given a 2-dimensional array where each column represents a sample, like:
scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

It should return a 2-dimensional array of the same shape, (3, 4):
[[ 0.09003057  0.00242826  0.01587624  0.33333333]
 [ 0.24472847  0.01794253  0.11731043  0.33333333]
 [ 0.66524096  0.97962921  0.86681333  0.33333333]]
 """
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/np.sum(np.exp(x), axis=0)
    """"
    dim = len(x.shape)
    if dim == 1:
        return np.exp(x)/np.sum(np.exp(x))
    if dim == 2:
        output = np.zeros(x.shape)
        ncol = x.shape[1]
        for i in xrange(ncol):
            col = x[:,[i]]
            out = np.exp(col) / np.sum(np.exp(col))
            output[:,[i]] = out
        return output
    """

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.01)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
plt.plot(x, softmax(scores).T, linewidth=2)
plt.xlabel("x")
plt.ylabel("softmax")
plt.show()

def test():
    """
    A few tests to make sure that the perceptron class performs as expected.
    Nothing should show up in the output if all the assertions pass.
    """
    def sum_almost_equal(array1, array2, tol = 1e-5):
        return np.sum(np.abs(array1 - array2)) < tol

    scores = np.array([1,2,3])
    prob1 = softmax(scores)
    assert sum_almost_equal(prob1, np.array([0.09003057, 0.24472847, 0.66524096]))
    
    scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])
    prob2 = softmax(scores)
    test2 = np.array([[ 0.09003057, 0.00242826, 0.01587624, 0.33333333],
        [ 0.24472847, 0.01794253, 0.11731043, 0.33333333],
        [ 0.66524096, 0.97962921, 0.86681333, 0.33333333]])
    assert sum_almost_equal(prob2, test2)

if __name__ == "__main__":
    test()