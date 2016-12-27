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

    def test_sigmoid(self):
        inputs, weights, bias = Input(), Input(), Input()

        f = Linear(inputs, weights, bias)
        g = Sigmoid(f)

        x = np.array([[-1., -2.], [-1, -2]])
        w = np.array([[2., -3], [2., -3]])
        b = np.array([-3., -5])

        feed_dict = {inputs: x, weights: w, bias: b}

        graph = topological_sort(feed_dict)
        output = forward_pass(g, graph)

        """
        Output should be:
        [[  1.23394576e-04   9.82013790e-01]
        [  1.23394576e-04   9.82013790e-01]]
        """
        expected = np.array([[1.23394576e-04, 9.82013790e-01], [1.23394576e-04, 9.82013790e-01]])
        self.assertTrue(np.allclose(expected, output))

    def test_mse(self):
        y, a = Input(), Input()
        cost = MSE(y, a) 

        y_ = np.array([1, 2, 3])
        a_ = np.array([4.5, 5, 10])

        feed_dict = {y: y_, a: a_}
        graph = topological_sort(feed_dict)
        # forward pass
        forward_pass_graph(graph)

        """
        Expected output

        23.4166666667
        """
        self.assertTrue(np.allclose(23.4166666667, cost.value))
        
if __name__ == '__main__':
    unittest.main()


