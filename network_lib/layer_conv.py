"""
Convolutional Layer
"""
from math import ceil
import tensorflow as tf
from layer_base import BaseLayer

class ConvolutionalLayer(BaseLayer):
    """
    Encapsulates a convolutional layer in a neural network

    Parameters:
    * input_shape: the tuple (input width, input height, input depth)
    * kernel_shape: the tuple (kernel width, kernel height, kernel depth)
    * kernel_stride_shape: the tuple (stride width, stride height)
    * kernel_padding_shape: the tuple (padding width, padding height)
    * kernel_padding_type: either 'SAME' or 'VALID'
    * activation_type: can only be 'relu' right now or None
    """
    def __init__(self, name, input_shape, kernel_shape, kernel_stride_shape, kernel_padding_shape, \
        kernel_padding_type, activation_type='relu'):
        super(ConvolutionalLayer, self).__init__(name)
        self.type = 'convolutional'
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.kernel_stride = [1, kernel_stride_shape[0], kernel_stride_shape[1], 1]
        self.kernel_padding = kernel_padding_shape
        self.kernel_padding_type = kernel_padding_type

        # Inputs, weights and biases
        self.mean = None
        self.stddev = None
        self.weights = None
        self.biases = None
        self.conv = None
        self.activation = None
        self.pool = None

        # Activation
        if activation_type not in [None, 'relu']:
            raise ValueError("activation_type must be in [None, 'relu']")
        self.activation_type = activation_type

        self.pool_type = None
        self.pool_kernel_shape = None
        self.pool_stride_shape = None
        self.pool_pad_type = None
        self.output_shape = self.conv_output_shape()

    def conv_output_shape(self):
        """
        Returns the shape of the convolution output
        """
        if self.kernel_padding_type == 'VALID':
            output_height = int(ceil((self.input_shape[0] - self.kernel_shape[0] + 2. * \
                self.kernel_padding[0]) / float(self.kernel_stride[1]) + 1))
            output_width = int(ceil((self.input_shape[1] - self.kernel_shape[1] + 2. * \
                self.kernel_padding[1]) / float(self.kernel_stride[2]) + 1))
        if self.kernel_padding_type == 'SAME':
            output_height = ceil(float(self.input_shape[0]) / float(self.kernel_stride[1]))
            output_width = ceil(float(self.input_shape[1]) / float(self.kernel_stride[2]))

        return (output_height, output_width, self.kernel_shape[2])

    def pool_output_shape(self):
        """
        Returns the shape of the convolution output
        """
        conv_shape = self.conv_output_shape()
        if self.pool_pad_type == 'VALID':
            output_height = int(ceil((conv_shape[0] - self.pool_kernel_shape[1] + 1) / \
                self.pool_stride_shape[1]))
            output_width = int(ceil((conv_shape[1] - self.pool_kernel_shape[2] + 1) / \
                self.pool_stride_shape[2]))
        if self.pool_pad_type == 'SAME':
            output_height = int(ceil(conv_shape[0] / self.pool_stride_shape[1]))
            output_width = int(ceil(conv_shape[1] / self.pool_stride_shape[2]))
        return (output_height, output_width, self.kernel_shape[2])

    def add_pooling(self, pool_type, pool_kernel_shape, pool_stride_shape, pool_pad_type):
        """
        Add pooling to the end of the convolution layer.

        Parameters:
        * pool_type: either 'max' or 'avg'
        * pool_kernel_shape: the tuple (pool kernel width, pool kernel height)
        * pool_stride_shape: the tuple (pool stride width, pool stride height)
        """
        if pool_type not in ['max', 'avg']:
            raise ValueError("pool_type must be 'max' or 'avg'.")

        if pool_pad_type not in ['VALID', 'SAME']:
            raise ValueError("pool_pad_type must be 'VALID' or 'SAME'.")

        self.pool_type = pool_type
        self.pool_kernel_shape = [1, pool_kernel_shape[0], pool_kernel_shape[1], 1]
        self.pool_stride_shape = [1, pool_stride_shape[0], pool_stride_shape[1], 1]
        self.pool_pad_type = pool_pad_type
        self.output_shape = self.pool_output_shape()

    def init_weights_and_biases(self, mean=0.0, stddev=1.0):
        """
        Randomly initialize weights and biases.  Weights will be initialized randomly
        from a truncated normal distribution. Biases will be initialized to zeros.

        Parameters:
        * mean: The mean of the normal distribution from which random weights will be drawn from
        * stddev: The standard deviation of the above normal distribution.
        """
        self.mean = mean
        self.stddev = stddev
        self.weights = tf.Variable(tf.truncated_normal((self.kernel_shape[0], \
            self.kernel_shape[1], self.input_shape[2], self.kernel_shape[2]), \
            mean, stddev), name=self.name + '_w')
        self.biases = tf.Variable(tf.zeros(self.kernel_shape[2]), name=self.name + '_b')

    def init_layer(self, inputs):
        """
        Initializes the inputs to the layer. Also set up the convolution, activation,
        and pooling, if any, and define the layer outputs.

        Parameters:
        * inputs: The data inputs into the layer
        """
        print("layer name:", self.name, "self.inputs", self.inputs)
        self.inputs = inputs
        self.conv = tf.nn.bias_add(tf.nn.conv2d(self.inputs, self.weights, self.kernel_stride, \
            self.kernel_padding_type), self.biases)
        self.outputs = self.conv
        if self.activation_type == 'relu':
            self.activation = tf.nn.relu(self.outputs)
            self.outputs = self.activation
        if self.pool_type == 'max':
            self.pool = tf.nn.max_pool(self.outputs, ksize=self.pool_kernel_shape, \
                strides=self.pool_stride_shape, padding=self.pool_pad_type)
            self.outputs = self.pool
        if self.pool_type == 'avg':
            self.pool = tf.nn.avg_pool(self.outputs, ksize=self.pool_kernel_shape, \
                strides=self.pool_stride_shape, padding=self.pool_pad_type)
            self.outputs = self.pool
