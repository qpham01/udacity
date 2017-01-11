"""
Base Layer
"""

class BaseLayer:
    """
    Abstract base class for neural network layer.  Every layer has a name, an input shape and
    an output shape.
    """
    def __init__(self, name):
        self.name = name
        self.input_shape = None
        self.output_shape = None
        self.inputs = None
        self.outputs = None
