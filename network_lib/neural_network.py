"""
Represents a neural network
"""
from time import time
from sklearn.utils import shuffle
import tensorflow as tf

class NeuralNetwork:
    """
    Encapsulates a neural network
    """
    def __init__(self, name, input_shape, output_count):
        self.name = name
        self.layers = []
        self.last_layer = None
        self.learning_rate = None
        self.cross_entropy = None
        self.loss_operation = None
        self.train_operation = None
        self.optimizer = None
        self.correct_prediction = None
        self.accuracy_operation = None
        self.inputs = None
        self.outputs = None
        self.labels = None
        input_dims = [None]
        for _, dim in enumerate(input_shape):
            input_dims.append(dim)
        self.input_placeholder = tf.placeholder(tf.float32, input_dims)
        self.label_placeholder = tf.placeholder(tf.int64, (None))
        self.output_count = output_count
        self.one_hot = tf.one_hot(self.label_placeholder, output_count)

    def add_layer(self, layer, mean, stddev):
        """
        Add a neural network layer

        Parameters:
        * layer: The layer to add to the network
        * mean: The mean of the normal distribution from which random weights will be drawn from
        * stddev: The standard deviation of the above normal distribution.
        """
        layer.init_weights_and_biases(mean, stddev)
        self.layers.append(layer)

    def define_network(self):
        """
        Initialize every layer with inputs and outputs
        """
        self.last_layer = None

        inputs = self.input_placeholder
        for layer in self.layers:
            if self.last_layer is not None:
                inputs = self.last_layer.outputs
            if layer.type == 'linear' and self.last_layer.type == 'convolutional':
                inputs = tf.reshape(inputs, [-1, layer.input_size])
            layer.init_layer(inputs)
            self.last_layer = layer
        self.outputs = self.last_layer.outputs

    def define_operations(self, learning_rate, optimizer):
        """
        Define various computations and operations for the network.

        Parameters:
        * learning_rate: The learning rate to use when optimizing for lower cost
        * optimizer:  The optimizer to use; must be in ['adam', 'gradient_descent']
        """
        optimizer_list = ['gradient_descent', 'adam']
        if optimizer not in optimizer_list:
            raise ValueError("optimizer must be in", optimizer_list)
        self.learning_rate = learning_rate
        # self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.outputs, self.one_hot)
        # Using sparse_softmax... means no need to use one_hot with logits, just the labels directly
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.outputs, \
            self.label_placeholder)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        if optimizer == 'gradient_descent':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        if optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_operation = self.optimizer.minimize(self.loss_operation)
        # self.correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.one_hot, 1))
        self.correct_prediction = tf.equal(tf.argmax(self.outputs, 1), self.label_placeholder)
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train_in_batches(self, sess, train_inputs, train_labels, batch_size):
        """
        Train network in batches of data
        """
        train_inputs, train_labels = shuffle(train_inputs, train_labels)
        num_examples = len(train_inputs)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = train_inputs[offset:end], train_labels[offset:end]
            sess.run(self.train_operation, feed_dict={self.input_placeholder: batch_x, \
                self.label_placeholder: batch_y})
        # Return cost of the last batch
        cost = sess.run(self.loss_operation, feed_dict={self.input_placeholder: batch_x, \
            self.label_placeholder: batch_y})
        return cost

    def evaluate_in_batches(self, sess, inputs, labels, batch_size):
        """
        Evaluate inputs in batches
        """
        num_examples = len(inputs)
        total_accuracy = 0

        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = inputs[offset:offset + batch_size], labels[offset:offset + \
                batch_size]
            accuracy = sess.run(self.accuracy_operation, \
                feed_dict={self.input_placeholder: batch_x, self.label_placeholder: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def train(self, sess, train_inputs, train_labels, train_epochs, batch_size):
        """
        Train the network with validation
        """

        print("Training...")
        print()

        time0 = time()
        for i in range(train_epochs):
            time1 = time()
            cost = self.train_in_batches(sess, train_inputs, train_labels, batch_size)

            print("Epoch:", '%04d' % (i), "Cost =", "{:.9f}".format(cost), \
                "Time elapsed: {:.2f}".format(time() - time1))
            print()

        print("Training Finished!")
        print("Total training time:", "{:.2f}".format(time() - time0))

    def train_with_validate(self, sess, train_inputs, train_labels, valid_inputs, valid_labels, \
        train_epochs, batch_size):
        """
        Train the network with validation
        """

        print("Training...")
        print()

        time0 = time()
        for i in range(train_epochs):
            time1 = time()
            cost = self.train_in_batches(sess, train_inputs, train_labels, batch_size)

            print("Epoch:", '%04d' % (i), "Cost =", "{:.9f}".format(cost), \
                "Time elapsed: {:.2f}".format(time() - time1))

            valid_accuracy = self.evaluate_in_batches(sess, valid_inputs, valid_labels, batch_size)
            print("Validation Accuracy = {:.3f}".format(valid_accuracy))
            print()
        print("Training Finished!")
        print("Total training time:", "{:.2f}".format(time() - time0))
