"""
Represents a neural network
"""
class NeuralNetwork:
    """
    Encapsulates a neural network
    """
    def __init__(self, name):
        self.name = name
        self.layers = []

    def add_layer(self, layer):
        """
        Add a neural network layer
        """
        self.layers.append(layer)

    def define_network(self, inputs):
        """
        Initialize every layer with inputs and outputs
        """
        last_layer = None
        for layer in self.layers:
            if last_layer is not None:
                inputs = last_layer.outputs
            layer.init_layer(inputs)
            last_layer = layer
        return last_layer.outputs
