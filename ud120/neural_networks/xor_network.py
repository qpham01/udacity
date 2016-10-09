# ----------
#
# In this exercise, you will create a network of perceptrons that can represent
# the XOR function, using a network structure like those shown in the previous
# quizzes.
#
# You will need to do two things:
# First, create a network of perceptrons with the correct weights
# Second, define a procedure EvalNetwork() which takes in a list of inputs and
# outputs the value of this network.
#
# ----------

import numpy as np
#from perceptron import Perceptron

class Perceptron:
    """
    This class models an artificial neuron with step activation function.
    """

    def __init__(self, weights = np.array([1]), threshold = 0):
        """
        Initialize weights and threshold based on input arguments. Note that no
        type-checking is being performed here for simplicity.
        """
        self.weights = weights
        self.threshold = threshold


    def activate(self, values):
        """
        Takes in @param values, a list of numbers equal to length of weights.
        @return the output of a threshold perceptron with given inputs based on
        perceptron weights and threshold.
        """
               
        # First calculate the strength with which the perceptron fires
        strength = np.dot(values,self.weights)
        
        # Then return 0 or 1 depending on strength compared to threshold  
        return int(strength > self.threshold)

            
# Part 1: Set up the perceptron network
# Want two perceptrons in two layers: an AND perceptron and an output-layer perceptron
# with 3 inpouts
# The AND perceptron has weigths of 0.5, 0.5, and threshold of 0.75
# The output-layer perceptron has weights of 1, -2, 1 and a threshold of 1
Network = [
    # input layer, declare input layer perceptrons here
    [ 0.5, 0.5 ], \
    # output node, declare output layer perceptron here
    [ 1, -2, 1 ]
]

# Part 2: Define a procedure to compute the output of the network, given inputs
def EvalNetwork(inputValues, Network):
    """
    Takes in @param inputValues, a list of input values, and @param Network
    that specifies a perceptron network. @return the output of the Network for
    the given set of inputs.
    """
    
    # YOUR CODE HERE
    pAnd = Perceptron(Network[0], 0.75) #, "and", False)
    pOutput = Perceptron(Network[1], 0.9) #, "output", False)
    andOutput = pAnd.activate(inputValues)
    OutputValue = pOutput.activate([inputValues[0], andOutput, inputValues[1]])
    
    # Be sure your output value is a single number
    return OutputValue


def test():
    """
    A few tests to make sure that the perceptron class performs as expected.
    """
    o1 = EvalNetwork(np.array([0,0]), Network)
    print "0 XOR 0 = 0?:", o1
    o2 = EvalNetwork(np.array([0,1]), Network)
    print "0 XOR 1 = 1?:", o2
    o3 = EvalNetwork(np.array([1,0]), Network)
    print "1 XOR 0 = 1?:", o3
    o4 = EvalNetwork(np.array([1,1]), Network)
    print "1 XOR 1 = 0?:", o4

if __name__ == "__main__":
    test()