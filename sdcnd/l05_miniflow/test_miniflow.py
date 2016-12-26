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

"""
    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
"""

if __name__ == '__main__':
    unittest.main()