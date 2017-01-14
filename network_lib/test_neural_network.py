"""
Test for neural network lib
"""
import unittest
import tensorflow as tf
from neural_network import NeuralNetwork
from layer_conv import ConvolutionalLayer
from layer_linear import LinearLayer
from mnist_data import get_mnist_data

class TestNeuralNetwork(unittest.TestCase):
    """
    Contains tests for neural network lib
    """
    def test_conv_layer_init(self):
        """
        Tests ConvolutionalLayer initialization
        """
        layer = ConvolutionalLayer('l1', (32, 32, 3), (5, 5, 10), (1, 1), (0, 0), 'VALID', 'relu')
        self.assertEqual('l1', layer.name)
        self.assertEqual((28, 28, 10), layer.output_shape)

        layer.add_pooling('max', (2, 2), (2, 2), 'VALID')
        self.assertEqual((14, 14, 10), layer.output_shape)

        # We have an input of shape 32x32x3 (HxWxD)
        # 20 filters of shape 8x8x3 (HxWxD)
        # A stride of 2 for both the height and width (S)
        # Valid padding of size 1 (P)
        layer = ConvolutionalLayer('l1', (32, 32, 3), (8, 8, 20), (2, 2), (1, 1), 'VALID', 'relu')
        self.assertEqual('l1', layer.name)
        self.assertEqual((14, 14, 20), layer.output_shape)

        layer.add_pooling('max', (2, 2), (2, 2), 'VALID')
        self.assertEqual((7, 7, 20), layer.output_shape)

    def test_network_lesson09(self):
        """
        Test initializing a neural NeuralNetwork
        """
        input_shape = (28, 28, 1)
        output_count = 10
        network = NeuralNetwork('convnet for MNIST', input_shape, output_count)

        # normal distribution parameters for random weights
        mean = 0.0
        stddev = 0.1

        # General convolution shapes and parameters common to all convolutional layers
        conv_stride_shape = (1, 1)
        conv_pad_shape = (0, 0)
        conv_pad_type = 'SAME'

        pool_stride_shape = (2, 2)
        pool_shape = (2, 2)
        pool_pad_type = 'SAME'

        activation = 'relu'

        # Kernel depths and sizes for each convolution layer
        depths = [32, 64, 128]
        kernel_shapes = [(5, 5, depths[0]), (5, 5, depths[1]), (5, 5, depths[2])]
        conv_layer_count = len(depths)

        # Expected values for assertions
        after_conv_output_shapes = [(28, 28, depths[0]), (14, 14, depths[1]), (7, 7, depths[2])]
        after_pool_output_shapes = [(14, 14, depths[0]), (7, 7, depths[1]), (4, 4, depths[2])]

        # Create convolutional layers
        conv = None
        for i in range(conv_layer_count):
            name = 'l{:d}'.format(i)
            if i > 0:
                input_shape = conv.output_shape
            conv = ConvolutionalLayer(name, input_shape, kernel_shapes[i], conv_stride_shape, \
                conv_pad_shape, conv_pad_type, activation)
            self.assertEqual(after_conv_output_shapes[i], conv.output_shape)
            conv.add_pooling('max', pool_shape, pool_stride_shape, pool_pad_type)
            self.assertEqual(after_pool_output_shapes[i], conv.output_shape)
            network.add_layer(conv, mean, stddev)

        # Create linear layers

        # Output sizes for linear layers
        linear_input_sizes = [4 * 4 * 128, 512]
        linear_output_sizes = [512, 10]
        linear_activations = ['tanh', None]

        for i, input_size in enumerate(linear_input_sizes):
            layer_index = i + conv_layer_count
            name = 'l{:d}'.format(layer_index)
            linear = LinearLayer(name, input_size, linear_output_sizes[i], linear_activations[i])
            network.add_layer(linear, mean, stddev)

        # MNIST classify 10 digits
        network.define_network()

        learning_rate = 0.001
        network.define_operations(learning_rate, 'gradient_descent')

        epochs = 10
        batch_size = 128
        saver = tf.train.Saver()
        (train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels) = \
            get_mnist_data()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            network.train_with_validate(sess, train_inputs, train_labels, valid_inputs, \
                valid_labels, epochs, batch_size)

            test_accuracy = network.evaluate_in_batches(sess, test_inputs, test_labels, batch_size)
            print("Test accuracy:", test_accuracy)

            saver.save(sess, 'convnet')
            print("Model saved")

    def test_network_lenet(self):
        """
        Test using the lenet5 architecture
        """
        input_shape = (32, 32, 1)
        output_count = 10
        network = NeuralNetwork('lenet5 for MNIST', input_shape, output_count)

        # normal distribution parameters for random weights
        mean = 0.0
        stddev = 0.1

        # General convolution shapes and parameters common to all convolutional layers
        conv_stride_shape = (1, 1)
        conv_pad_shape = (0, 0)
        conv_pad_type = 'VALID'

        pool_stride_shape = (2, 2)
        pool_shape = (2, 2)
        pool_pad_type = 'VALID'

        activation = 'relu'

        # Kernel depths and sizes for each convolution layer
        depths = [6, 16]
        kernel_shapes = [(5, 5, depths[0]), (5, 5, depths[1])]
        conv_layer_count = len(depths)

        # Create convolutional layers
        conv = None
        for i in range(conv_layer_count):
            name = 'l{:d}'.format(i)
            if i > 0:
                input_shape = conv.output_shape
            conv = ConvolutionalLayer(name, input_shape, kernel_shapes[i], conv_stride_shape, \
                conv_pad_shape, conv_pad_type, activation)
            conv.add_pooling('max', pool_shape, pool_stride_shape, pool_pad_type)
            network.add_layer(conv, mean, stddev)

        # Linear layer dimensions
        linear_input_sizes = [400, 120, 84]
        linear_output_sizes = [120, 84, 10]
        linear_activations = ['relu', 'relu', None]

        # Create linear layers
        for i, input_size in enumerate(linear_input_sizes):
            layer_index = i + conv_layer_count
            name = 'l{:d}'.format(layer_index)
            linear = LinearLayer(name, input_size, linear_output_sizes[i], linear_activations[i])
            linear.init_weights_and_biases(mean, stddev)
            network.add_layer(linear, mean, stddev)

        network.define_network()

        learning_rate = 0.001
        network.define_operations(learning_rate, 'adam')

        # Prepare data
        (train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels) = \
            get_mnist_data(padding=(2, 2))

        epochs = 10
        batch_size = 128
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            network.train_with_validate(sess, train_inputs, train_labels, valid_inputs, \
                valid_labels, epochs, batch_size)

            test_accuracy = network.evaluate_in_batches(sess, test_inputs, test_labels, batch_size)
            print("Test accuracy:", test_accuracy)

            saver.save(sess, 'lenet')
            print("Model saved")

if __name__ == '__main__':
    unittest.main()
