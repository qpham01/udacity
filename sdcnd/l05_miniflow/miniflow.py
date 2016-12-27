"""
Encapsulates a neuron in a neural network
"""
import math
import operator
import numpy as np
from functools import reduce

"""
Modify Linear#forward so that it linearly transforms
input matrices, weights matrices and a bias vector to
an output.
"""

class Layer:
    """
    A layer in a neural network
    """
    def __init__(self, inbound_layers=[]):
        self.inbound_layers = inbound_layers
        self.value = None
        self.outbound_layers = []
        self.gradients = {}
        for layer in inbound_layers:
            layer.outbound_layers.append(self)

    def forward(self):
        """
        Forward propagate a neural network layer
        """
        raise NotImplementedError

    def backward(self):
        """
        Every layer that uses this class as a base class will
        need to define its own `backward` method.
        """
        raise NotImplementedError

class Input(Layer):
    """
    While it may be strange to consider an input a layer when
    an input is only an individual node in a layer, for the sake
    of simpler code we'll still use Layer as the base class.

    Think of Input as collating many individual input nodes into
    a Layer.
    """
    def __init__(self):
        # An Input layer has no inbound layers,
        # so no need to pass anything to the Layer instantiator
        Layer.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated.
        pass

    def backward(self):
        # An Input layer has no inputs so the gradient (derivative)
        # is zero.
        # The key, `self`, is reference to this object.
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbound_layers:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1

class Linear(Layer):
    def __init__(self, inbound_layer, weights, bias):
        # Notice the ordering of the input layers passed to the
        # Layer constructor.
        Layer.__init__(self, [inbound_layer, weights, bias])

    def forward(self):
        """
        Set the value of this layer to the linear transform output.

        Your code goes here!
        """
        inputs = self.inbound_layers[0]
        weights = self.inbound_layers[1]
        bias = self.inbound_layers[2]

        self.value = bias.value + np.matmul(inputs.value, weights.value)

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the inbound_layers.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_layers}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_layers:
            # Get the partial of the cost with respect to this layer.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this layer's inputs.
            self.gradients[self.inbound_layers[0]] += np.dot(grad_cost, \
                self.inbound_layers[1].value.T)
            # Set the partial of the loss with respect to this layer's weights.
            self.gradients[self.inbound_layers[1]] += np.dot(self.inbound_layers[0].value.T, \
                grad_cost)
            # Set the partial of the loss with respect to this layer's bias.
            self.gradients[self.inbound_layers[2]] += np.sum(grad_cost, axis=0, keepdims=False)

class Sigmoid(Layer):
    """
    You need to fix the `_sigmoid` and `forward` methods.
    """
    def __init__(self, layer):
        Layer.__init__(self, [layer])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.

        Return the result of the sigmoid function.

        Your code here!
        """
        return 1. / (1. + np.exp(-x))

    def forward(self):
        """
        Set the value of this layer to the result of the
        sigmoid function, `_sigmoid`.

        Your code here!
        """
        # This is a dummy value to prevent numpy errors
        # if you test without changing this method.
        inputs = self.inbound_layers[0].value
        self.value = self._sigmoid(inputs)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_layers}

        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_layers:
            # Get the partial derivative of the cost with respect to this layer.
            grad_cost = n.gradients[self]
            """
            TODO: Your code goes here!

            Set the gradients property to the gradients with respect to each input.

            NOTE: See the Linear layer and MSE layer for examples.
            """

            # Calculate the sigmoid of input X as assign to self.value
            # Note: this is already done in the forward pass, but is included here for reference.
            # X = self.inbound_layers[0].value
            # z = self._sigmoid(X)

            # Calculate the derivative of the sigmoid.
            ds = self.value * (1. - self.value)

            # Apply recursive chain rule of iteratively multiplying partial derivatives
            # for this layer
            dx = grad_cost * ds

            # add gradient
            self.gradients[self.inbound_layers[0]] += dx

class MSE(Layer):
    """
    Encapsulate means square EOFError
    """
    def __init__(self, y, a):
        """
        The mean squared error cost function.
        Should be used as the last layer for a network.
        """
        # Call the base class' constructor.
        Layer.__init__(self, [y, a])

        self.m = None
        self.diff = None

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
        y = self.inbound_layers[0].value.reshape(-1, 1)
        a = self.inbound_layers[1].value.reshape(-1, 1)
        self.m = y.shape[0]

        self.diff = y - a
        self.value = np.mean(self.diff**2)

    def backward(self):
        """
        Calculates the gradient of the cost.

        This is the final layer of the network so outbound layers
        are not a concern.
        """
        self.gradients[self.inbound_layers[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_layers[1]] = (-2 / self.m) * self.diff

def topological_sort(feed_dict):
    """
    Sort the layers in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective
    value feed to that Layer.

    Returns a list of sorted layers.
    """

    input_layers = [n for n in feed_dict.keys()]

    G = {}
    layers = [n for n in input_layers]
    while len(layers) > 0:
        n = layers.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_layers:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            layers.append(m)

    L = []
    S = set(input_layers)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_layers:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def forward_pass_graph(graph):
    """
    Performs a forward pass through a list of sorted Layers.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()

def forward_pass(output_layer, sorted_layers):
    """
    Performs a forward pass through a list of sorted Layers.

    Arguments:

        `output_layer`: A Layer in the graph, should be the output layer (have no outgoing edges).
        `sorted_layers`: a topologically sorted list of layers.

    Returns the output layer's value
    """

    for n in sorted_layers:
        n.forward()

    return output_layer.value

def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Layers.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in graph[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=0.01):
    """
    Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` Layers representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # TODO: update all the `trainables` with SGD
    # You can access and assign the value of a trainable with `value` attribute.
    # Example:
    # for t in trainables:
    #   t.value = your implementation here
    for t in trainables:
        t.value = t.value - learning_rate * t.gradients[t]

