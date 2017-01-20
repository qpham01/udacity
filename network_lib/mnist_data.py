"""
Prepare MNIST data for test
"""
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import numpy as np

def get_mnist_data(padding=(0, 0)):
    """
    Returns formatted input data
    """
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    train_inputs, train_labels = mnist.train.images, mnist.train.labels
    valid_inputs, valid_labels = mnist.validation.images, mnist.validation.labels
    test_inputs, test_labels = mnist.test.images, mnist.test.labels

    assert len(train_inputs) == len(train_labels)
    assert len(valid_inputs) == len(valid_labels)
    assert len(test_inputs) == len(test_labels)

    print()
    print("Image Shape: {}".format(train_inputs[0].shape))
    print()
    print("Training Set:   {} samples".format(len(train_inputs)))
    print("Validation Set: {} samples".format(len(valid_inputs)))
    print("Test Set:       {} samples".format(len(test_inputs)))
    print("Test Label Shape: {}".format(test_labels.shape))

    # Pad images with 0s
    train_inputs = np.pad(train_inputs, ((0, 0), padding, padding, (0, 0)), 'constant')
    valid_inputs = np.pad(valid_inputs, ((0, 0), padding, padding, (0, 0)), 'constant')
    test_inputs = np.pad(test_inputs, ((0, 0), padding, padding, (0, 0)), 'constant')

    print("Updated Image Shape: {}".format(train_inputs[0].shape))
    train_inputs, train_labels = shuffle(train_inputs, train_labels)

    return (train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels)
