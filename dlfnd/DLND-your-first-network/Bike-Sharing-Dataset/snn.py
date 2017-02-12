'''
Sample neural network, defined as follows:

input: 0.1 ---> weight: 0.4  |
                              ---> sigmoid node ---> weight: 0.1 ---> sigmoid node ---> output
input: 0.3 ---> weight: -0.2 |
'''
import numpy as np

class SampleNeuralNetwork(object):
    '''
    First Neural Network.
    '''
    def __init__(self, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = 2
        self.hidden_nodes = 1
        self.output_nodes = 1

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, \
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, \
                                       (self.output_nodes, self.hidden_nodes))
        self.learn_rate = learning_rate

        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = sigmoid

        self.hidden_inputs = None
        self.hidden_outputs = None
        self.final_inputs = None
        self.final_outputs = None
        self.output_errors = None
        self.hidden_errors = None
        self.hidden_grad = None
        self.output_grad = None

    def train(self, inputs_list, target):
        ''' Train the neural network by updating weights via back propagation '''
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        #targets = np.array(targets_list, ndmin=2).T

        #### Implement the forward pass here ####
        ### Forward pass ###
        # Hidden layer
        self.hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        self.hidden_outputs = sigmoid(self.hidden_inputs)

        # Output layer
        self.final_inputs = np.dot(self.weights_hidden_to_output, self.hidden_outputs)
        self.final_outputs = sigmoid(self.final_inputs)
        print(self.final_outputs)
        #### Implement the backward pass here ####
        ### Backward pass ###

        # Output error
        # error = (output - target) * f'(W * x) = (output - target) * sigmoid_deriv(W * x)
        print("Targets", target)

        self.output_errors = np.dot((target - self.final_outputs), sigmoid_deriv(self.final_inputs))
        self.output_grad = self.learn_rate * self.output_errors * self.hidden_outputs

        # Backpropagated error
        self.hidden_errors = self.output_errors * self.weights_hidden_to_output * \
            sigmoid_deriv(self.hidden_inputs)
        self.hidden_grad = self.learn_rate * self.hidden_errors * inputs

        # Update the weights
        # self.weights_hidden_to_output += - self.hidden_grad
        # self.weights_input_to_hidden += - self.learn_rate * self.output_errors

    def run(self, inputs_list):
        ''' Run a forward pass through the network '''
        inputs = np.array(inputs_list, ndmin=2).T

        #### Implement the forward pass here ####
        # Hidden layer
        self.hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        self.hidden_outputs = self.activation_function(self.hidden_inputs)

        # Output layer
        self.final_inputs = np.dot(self.weights_hidden_to_output, self.hidden_outputs)
        self.final_outputs = self.final_inputs

        return self.final_outputs

def sigmoid(x):
    ''' Calculate the sigmoid function for some input '''
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    ''' Calculated the derviative of the sigmoid for some input '''
    return sigmoid(x) * (1 - sigmoid(x))

