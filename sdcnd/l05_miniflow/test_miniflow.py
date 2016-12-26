import unittest
from miniflow import *

class TestMiniflow(unittest.TestCase):

    def test_add_2(self):
        x, y = Input(), Input()

        f = Add([x, y])

        feed_dict = {x: 10, y: 5}

        sorted_neurons = topological_sort(feed_dict)
        output = forward_pass(f, sorted_neurons)

        self.assertEqual(15, output)

    def test_add_n(self):
        x, y, z = Input(), Input(), Input()

        f = Add([x, y, z])

        feed_dict = {x: 10, y: 5, z: 4}

        sorted_neurons = topological_sort(feed_dict)
        output = forward_pass(f, sorted_neurons)

        self.assertEqual(19, output)

    def test_mul_n(self):
        x, y, z = Input(), Input(), Input()

        f = Mul([x, y, z])

        feed_dict = {x: 10, y: 5, z: 4}

        sorted_neurons = topological_sort(feed_dict)
        output = forward_pass(f, sorted_neurons)

        self.assertEqual(200, output)

    def test_linear(self):
        x, y, z = Input(), Input(), Input()
        inputs = [x, y, z]

        weight_x, weight_y, weight_z = Input(), Input(), Input()
        weights = [weight_x, weight_y, weight_z]

        bias = Input()

        f = Linear(inputs, weights, bias)

        feed_dict = {
            x: 6,
            y: 14,
            z: 3,
            weight_x: 0.5,
            weight_y: 0.25,
            weight_z: 1.4,
            bias: 2
        }

        graph = topological_sort(feed_dict)
        output = forward_pass(f, graph)

        self.assertEqual(12.7, output)

if __name__ == '__main__':
    unittest.main()