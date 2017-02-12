''' Test first neural network '''

import unittest
import numpy as np
from snn import SampleNeuralNetwork

class TestNeuralNetworks(unittest.TestCase):
    '''
    Test neural networks
    '''
    def test_train(self):
        ''' Test training '''
        learning_rate = 0.5

        snn = SampleNeuralNetwork(learning_rate)

        input_list = [0.1, 0.3]
        target = 1.0

        snn.weights_input_to_hidden = np.array([0.4, -0.2])
        snn.weights_hidden_to_output = np.array([0.1])

        snn.train(input_list, target)

        print("Hidden outputs ", snn.hidden_outputs)
        print("Final inputs   ", snn.final_inputs)
        print("Final_outputs  ", snn.final_outputs)
        print("Output error   ", snn.output_errors)
        print("Hidden error   ", snn.hidden_errors)
        print("Output gradient", snn.output_grad)
        print("Hidden gradient", snn.hidden_grad)

        self.assertAlmostEqual(-0.02, snn.hidden_inputs[0], 5)
        self.assertAlmostEqual(0.495, snn.hidden_outputs[0], 5)
        self.assertAlmostEqual(0.0495, snn.final_inputs, 5)
        self.assertAlmostEqual(0.5123725, snn.final_outputs, 5)
        self.assertAlmostEqual(0.1218322, snn.output_errors, 5)
        self.assertAlmostEqual(0.00304394, snn.hidden_errors[0], 3)
        # self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()
