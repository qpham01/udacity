import unittest
import numpy as np
from miniflow import *

class TestMiniflow(unittest.TestCase):
    """
    Encapsulate testing of miniflow
    """
    def test_linear(self):
        """
        Test linear layer forward propagation
        """
        inputs, weights, bias = Input(), Input(), Input()

        f = Linear(inputs, weights, bias)

        x = np.array([[-1., -2.], [-1, -2]])
        w = np.array([[2., -3], [2., -3]])
        b = np.array([-3., -5])

        feed_dict = {inputs: x, weights: w, bias: b}

        graph = topological_sort(feed_dict)
        output = forward_pass(f, graph)

        """
        Output should be: [[-9., 4.], [-9., 4.]]
        """
        expected = np.array([[-9., 4.], [-9., 4.]])
        self.assertTrue(np.array_equal(expected, output))

if __name__ == '__main__':
    unittest.main()
