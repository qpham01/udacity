# ----------
#
# In this exercise, you will update the perceptron class so that it can update
# its weights.
#
# Finish writing the update() method so that it updates the weights according
# to the perceptron update rule.
# 
# ----------

import numpy as np


class Perceptron:
    """
    This class models an artificial neuron with step activation function.
    """

    def __init__(self, weights = np.array([1]), threshold = 0, name="perceptron", verbose=True):
        """
        Initialize weights and threshold based on input arguments. Note that no
        type-checking is being performed here for simplicity.
        """
        self.name = name
        self.weights = weights
        self.threshold = threshold
        self.verbose = verbose

    def strength(self, values):
        # First calculate the strength with which the perceptron fires
        strength = np.dot(values,self.weights)
        if (self.verbose):
            print "name: ", self.name, " strength: ", strength
        return strength

    def activate(self, values):
        """
        Takes in @param values, a list of numbers equal to length of weights.
        @return the output of a threshold perceptron with given inputs based on
        perceptron weights and threshold.
        """
        strength = self.strength(values)

        # Then return 0 or 1 depending on strength compared to threshold  
        output = int(strength > self.threshold)
        if (self.verbose):
            print "name: ", self.name, " output: ", output
        return output

    def update(self, values, train, eta=.1):
        """
        Takes in a 2D array @param values consisting of a LIST of inputs and a
        1D array @param train, consisting of a corresponding list of expected
        outputs. Updates internal weights according to the perceptron training
        rule using these values and an optional learning rate, @param eta.
        """
        """
        new_w_i = w_i + delta_w_i
        delta_w_i = learning_rate * (training_output - current_output) * training_input
        delta_w_i = learning_rate * (y_i - y_hat_i) * x_i
        y_hat_i = sum(dot(w_i, x_i)) > threshold
        y_i y_hat_i  y_i - y_hat_i
        0   0        0 
        0   1       -1
        1   0        1
        1   1        0
        """
        # YOUR CODE HERE
        for i,e in zip(train,values):
            diff = np.subtract(i,self.activate(e))
            e = e.astype(np.float64)
            delta = np.multiply(eta*diff,e)
            self.weights = np.add(delta , self.weights.astype(np.float64))
        
        """"
        # TODO: for each data point...
        print train.shape
        n = len(train)
        y_hat = []
        #each row in values (x_i, i.e. values[i]) is a set of inputs
        #that produces the scalar training output train[i] (y_i)
        for i in xrange(n):
            # TODO: obtain the neuron's prediction for that point
            y_hat.append(self.activate(values[i]))
            # TODO: update self.weights based on prediction accuracy, learning
            # rate and input value
            #print "i ", i, " train ", train[i], " inputs ", values[i] 
            dot_product = np.dot((train[i] - y_hat[i]), values[i])
            #print "i ", i, " dot product ", dot_product
            self.weights = self.weights + eta * dot_product
            print "updated_weights: ", self.weights
        """
def test():
    """
    A few tests to make sure that the perceptron class performs as expected.
    Nothing should show up in the output if all the assertions pass.
    """
    def sum_almost_equal(array1, array2, tol = 1e-6):
        return sum(abs(array1 - array2)) < tol

    p1 = Perceptron(np.array([1,1,1]),0)
    p1.update(np.array([[2,0,-3]]), np.array([1]))
    assert sum_almost_equal(p1.weights, np.array([1.2, 1, 0.7]))

    p2 = Perceptron(np.array([1,2,3]),0)
    p2.update(np.array([[3,2,1],[4,0,-1]]),np.array([0,0]))
    assert sum_almost_equal(p2.weights, np.array([0.7, 1.8, 2.9]))

    p3 = Perceptron(np.array([3,0,2]),0)
    p3.update(np.array([[2,-2,4],[-1,-3,2],[0,2,1]]),np.array([0,1,0]))
    assert sum_almost_equal(p3.weights, np.array([2.7, -0.3, 1.7]))

if __name__ == "__main__":
    test()