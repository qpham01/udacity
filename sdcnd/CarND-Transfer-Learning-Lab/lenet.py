"""
Test for neural network lib
"""
from neural_network import NeuralNetwork
from layer_conv import ConvolutionalLayer
from layer_linear import LinearLayer

def lenet_network(name, input_shape, output_count):
    """
    Run the lenet5 architecture on some data.
    """
    network = NeuralNetwork(name, input_shape, output_count)

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

    return network
