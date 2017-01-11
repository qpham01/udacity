"""
Test for neural network lib
"""
import unittest
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from neural_network import NeuralNetwork
from layer_conv import ConvolutionalLayer

class TestNeuralNetwork(unittest.TestCase):
    """
    Contains tests for neural network lib
    """
    def test_conv_layer_init(self):
        """
        Tests ConvolutionalLayer initialization
        """
        layer = ConvolutionalLayer('l1', (32, 32, 3), (5, 5, 10), (1, 1), (0, 0), 'VALID', \
            'relu')
        self.assertEqual('l1', layer.name)
        self.assertEqual((28, 28, 10), layer.output_shape)

        layer.add_pooling('max', (2, 2), (2, 2), 'VALID')
        self.assertEqual((14, 14, 10), layer.output_shape)

    def test_network(self):
        """
        Test initializing a neural NeuralNetwork
        """
        network = NeuralNetwork('lenet5')

        # normal distribution parameters for random weights
        mean = 0.0
        stddev = 1.0

        # General convolution shapes and parameters common to all convolutional layers
        conv_stride_shape = (1, 1)
        pad_shape = (0, 0)
        pad_type = 'SAME'
        pool_pad_type = 'SAME'
        activation = 'relu'
        pool_shape = (2, 2)
        pool_stride_shape = (2, 2)

        # Layer 1
        input_shape1 = (28, 28, 1)
        depth1 = 32
        kernel_shape1 = (5, 5, depth1)
        conv1 = ConvolutionalLayer('l1', input_shape1, kernel_shape1, conv_stride_shape, \
            pad_shape, pad_type, activation)
        self.assertEqual((28, 28, depth1), conv1.output_shape)
        conv1.add_pooling('max', pool_shape, pool_stride_shape, pool_pad_type)
        self.assertEqual((14, 14, depth1), conv1.output_shape)
        conv1.init_weights_and_biases(mean, stddev)

        # Layer 2
        depth2 = 64
        kernel_shape2 = (5, 5, depth2)
        conv2 = ConvolutionalLayer('l2', conv1.output_shape, kernel_shape2, conv_stride_shape, \
            pad_shape, pad_type, activation)
        self.assertEqual((14, 14, depth2), conv2.output_shape)
        conv2.add_pooling('max', pool_shape, pool_stride_shape, pool_pad_type)
        self.assertEqual((7, 7, depth2), conv2.output_shape)
        conv2.init_weights_and_biases(mean, stddev)

        # Layer 3
        depth3 = 128
        kernel_shape3 = (5, 5, depth3)
        conv3 = ConvolutionalLayer('l3', conv2.output_shape, kernel_shape3, conv_stride_shape, \
            pad_shape, pad_type, activation)
        self.assertEqual((7, 7, depth3), conv3.output_shape)
        conv3.add_pooling('max', pool_shape, pool_stride_shape, pool_pad_type)
        self.assertEqual((4, 4, depth3), conv3.output_shape)
        conv3.init_weights_and_biases(mean, stddev)

        # Add layers to network in sequence
        network.add_layer(conv1)
        network.add_layer(conv2)
        network.add_layer(conv3)

        # MNIST classify 10 digits
        n_classes = 10

        features = tf.placeholder("float", [None, 28, 28, 1])
        labels = tf.placeholder("float", [None, n_classes])

        logits = network.define_network(features)

        mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

if __name__ == '__main__':
    unittest.main()
