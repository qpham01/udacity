import numpy as np
from perceptron import Perceptron

def strength_network():
    p1 = Perceptron(np.array([1,1,-5]), 0, "p1")
    p2 = Perceptron(np.array([3,-4,2]), 0, "p2")
    p3 = Perceptron(np.array([2,-1]), 0, "p3")

    input = [1,2,3]
    p3_input = [p1.strength(input), p2.strength(input)]
    output = p3.strength(p3_input)
    return output

output = strength_network()
print "strength network output: ", output

"""
Network of form [[input, input], [[3,2],[-1,4],[3,-5]], [[1,2,-1]]]
i, j
3*i + 2*j + (-1*i + 4*j) * 2 + (3*i - 5*j) * -1
=3*i + 2*j -2*i + 8*j - 3*i + 5*j
=-2*i 
""" 