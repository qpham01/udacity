"""
Train and validate the lenet network on the CFAR-10 dataset
"""
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from lenet import lenet_network

def run_cifar():
    """
    Run the CIFAR data
    """
    # Prepare data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print("Training Set:      {} samples".format(len(x_train)))
    print("Test Set:          {} samples".format(len(x_test)))
    print("Test Labels Shape: {}".format(y_test.shape))

    y_train = np.reshape(y_train, len(y_train))
    y_test = np.reshape(y_test, len(y_test))

    print("Test Labels Reshape: {}".format(y_test.shape))

    epochs = 10
    batch_size = 128

    input_shape = (32, 32, 3)
    output_count = 10

    network = lenet_network('LeNet5 for CFAR-10', input_shape, output_count)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        network.train(sess, x_train, y_train, epochs, batch_size)

        test_accuracy = network.evaluate_in_batches(sess, x_test, y_test, batch_size)
        print("Test accuracy:", test_accuracy)

        saver.save(sess, 'saves/lenet_cfar10')
        print("Model saved")

run_cifar()
