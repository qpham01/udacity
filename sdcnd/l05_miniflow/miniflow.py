"""
Encapsulates a neuron in a neural network
"""
import math
import operator
from functools import reduce

class Neuron:
    def __init__(self, inbound_neurons=[]):
        # Neuron from which this Neuron receives values
        self.inbound_neurons = inbound_neurons
        # Neuron to which this Neuron passes values
        self.outbound_neurons = []
        # A calculated value
        self.value = None
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_neurons:
            n.outbound_neurons.append(self)

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_neurons` and
        store the result in self.value.
        """
        raise NotImplementedError

    def backward(self):
        """
        Backward propagation.

        You'll compute this later.
        """
        raise NotImplementedError

class Input(Neuron):
    def __init__(self):
        # An Input neuron has no inbound neurons,
        # so no need to pass anything to the Neuron instantiator.
        Neuron.__init__(self)

    # NOTE: Input neuron is the only node where the value
    # may be passed as an argument to forward().
    #
    # All other neuron implementations should get the value
    # of the previous neuron from self.inbound_neurons
    #
    # Example:
    # val0 = self.inbound_neurons[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value

class Add(Neuron):

    def forward(self):
        """
        You'll be writing code here in the next quiz!
        """
        values = [x.value for x in self.inbound_neurons]
        self.value = math.fsum(values)

class Mul(Neuron):

    def forward(self):
        """
        You'll be writing code here in the next quiz!
        """
        values = [x.value for x in self.inbound_neurons]
        self.value = reduce(operator.mul, values, 1)


"""
No need to change anything below here!
"""
def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective
    value feed to that node.

    Returns a list of sorted nodes.
    """

    input_neurons = [n for n in feed_dict.keys()]

    G = {}
    neurons = [n for n in input_neurons]
    while len(neurons) > 0:
        n = neurons.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_neurons:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            neurons.append(m)

    L = []
    S = set(input_neurons)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_neurons:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_neuron, sorted_neurons):
    """
    Performs a forward pass through a list of sorted neurons.

    Arguments:

        `output_neuron`: A neuron in the graph, should be the output neuron (have no outgoing
             edges).
        `sorted_neurons`: a topologically sorted list of neurons.

    Returns the output neuron's value
    """

    for n in sorted_neurons:
        n.forward()

    return output_neuron.value
