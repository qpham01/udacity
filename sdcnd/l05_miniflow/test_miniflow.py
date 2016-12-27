import unittest
import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
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

    def test_gradients(self):
        X, W, b = Input(), Input(), Input()
        y = Input()
        f = Linear(X, W, b)
        a = Sigmoid(f)
        cost = MSE(y, a)

        X_ = np.array([[-1., -2.], [-1, -2]])
        W_ = np.array([[2.], [3.]])
        b_ = np.array([-3.])
        y_ = np.array([1, 2])

        feed_dict = {
            X: X_,
            y: y_,
            W: W_,
            b: b_,
        }

        graph = topological_sort(feed_dict)
        forward_and_backward(graph)
        # return the gradients for each Input
        gradients = [t.gradients[t] for t in [X, y, W, b]]

        """
        Expected output

        [array([[ -3.34017280e-05,  -5.01025919e-05],
            [ -6.68040138e-05,  -1.00206021e-04]]), array([[ 0.9999833],
            [ 1.9999833]]), array([[  5.01028709e-05],
            [  1.00205742e-04]]), array([ -5.01028709e-05])]
        """
        expected = []
        expected.append(np.array([[-3.34017280e-05, -5.01025919e-05], [-6.68040138e-05, \
            -1.00206021e-04]]))
        expected.append(np.array([[0.9999833], [1.9999833]]))
        expected.append(np.array([[5.01028709e-05], [1.00205742e-04]]))
        expected.append(np.array([-5.01028709e-05]))
        for ex, gradient in zip(expected, gradients):
            self.assertTrue(np.allclose(ex, gradient))

    def test_sgd(self):
        # Load data
        data = load_boston()
        X_ = data['data']
        y_ = data['target']

        # Normalize data
        X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

        n_features = X_.shape[1]
        n_hidden = 10
        W1_ = np.random.randn(n_features, n_hidden)
        b1_ = np.zeros(n_hidden)
        W2_ = np.random.randn(n_hidden, 1)
        b2_ = np.zeros(1)

        # Neural network
        X, y = Input(), Input()
        W1, b1 = Input(), Input()
        W2, b2 = Input(), Input()

        l1 = Linear(X, W1, b1)
        s1 = Sigmoid(l1)
        l2 = Linear(s1, W2, b2)
        cost = MSE(y, l2)

        feed_dict = {
            X: X_,
            y: y_,
            W1: W1_,
            b1: b1_,
            W2: W2_,
            b2: b2_
        }

        epochs = 10
        # Total number of examples
        m = X_.shape[0]
        batch_size = 11
        steps_per_epoch = m // batch_size

        graph = topological_sort(feed_dict)
        trainables = [W1, b1, W2, b2]

        print("Total number of examples = {}".format(m))

        # Step 4
        last_loss = 0
        for i in range(epochs):
            loss = 0
            for j in range(steps_per_epoch):
                # Step 1
                # Randomly sample a batch of examples
                X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

                # Reset value of X and y Inputs
                X.value = X_batch
                y.value = y_batch

                # Step 2
                forward_and_backward(graph)

                # Step 3
                sgd_update(trainables)

                loss += graph[-1].value

            #print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))
            last_loss = loss/steps_per_epoch
        self.assertTrue(last_loss < 20.0)

if __name__ == '__main__':
    unittest.main()
