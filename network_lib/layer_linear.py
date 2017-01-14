"""
Linear Layer
"""
import tensorflow as tf
from layer_base import BaseLayer

class LinearLayer(BaseLayer):
    """
    Encapsulates a linear layer in a neural network

    Parameters:
    * input_size: the number of inputs
    * output_size: the number of outputs
    * activation_type: can only be 'tanh' right now
    """
    def __init__(self, name, input_size, output_size, activation_type='tanh'):
        super(LinearLayer, self).__init__(name)
        self.type = 'linear'
        self.input_size = input_size
        self.output_size = output_size
        self.good_activation_types = [None, 'relu', 'tanh']
        if activation_type not in self.good_activation_types:
            raise ValueError("activation type must be in", self.good_activation_types)
        self.activation_type = activation_type

        # Weights and biases
        self.weights = None
        self.biases = None

    def init_weights_and_biases(self, mean=0.0, stddev=1.0):
        """
        Randomly initialize weights and biases.  Weights will be initialized randomly
        from a truncated normal distribution. Biases will be initialized to zeros.

        Parameters:
        * mean: The mean of the normal distribution from which random weights will be drawn from
        * stddev: The standard deviation of the above normal distribution.
        """
        self.weights = tf.Variable(tf.truncated_normal([self.input_size, self.output_size], \
            mean, stddev), name=self.name + '_w')
        self.biases = tf.Variable(tf.zeros(self.output_size), self.name + '_b')

    def init_layer(self, inputs):
        """
        Initializes the inputs to the layer and set up the matrix multiplication.

        Parameters:
        * inputs: The data inputs into the layer
        """
        self.inputs = inputs
        self.outputs = tf.add(tf.matmul(self.inputs, self.weights), self.biases)
        if self.activation_type == 'relu':
            self.outputs = tf.nn.relu(self.outputs)
        if self.activation_type == 'tanh':
            self.outputs = tf.nn.tanh(self.outputs)
        