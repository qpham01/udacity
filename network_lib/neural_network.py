"""
Represents a neural network
"""
from time import time
import tensorflow as tf

class NeuralNetwork:
    """
    Encapsulates a neural network
    """
    def __init__(self, name):
        self.name = name
        self.layers = []
        self.last_layer = None
        self.learning_rate = None
        self.cost = None
        self.optimizer = None
        self.inputs = None
        self.outputs = None
        self.labels = None

    def add_layer(self, layer):
        """
        Add a neural network layer
        """
        self.layers.append(layer)

    def define_network(self, inputs):
        """
        Initialize every layer with inputs and outputs
        """
        self.last_layer = None

        for layer in self.layers:
            if self.last_layer is not None:
                inputs = self.last_layer.outputs
                print("inputs", inputs)
            if layer.type == 'linear' and self.last_layer.type == 'convolutional':
                inputs = tf.reshape(inputs, [-1, layer.input_size])
                print("flattened inputs", inputs)
            layer.init_layer(inputs)
            self.last_layer = layer
        self.outputs = self.last_layer.outputs

    def optimize_softmax_cross_entropy(self, learning_rate, optimizer, label_placeholder):
        """
        Define cost softmax as cross-entropy
        """
        if optimizer not in ['gradient_descent', 'adam']:
            raise ValueError("optimizer must be in ['gradient_descent', 'adam]")
        self.learning_rate = learning_rate
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
            self.last_layer.outputs, label_placeholder))
        if optimizer == 'gradient_descent':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) \
                .minimize(self.cost)
        if optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def train(self, training_epochs, next_batch, data_size, batch_size, input_placeholder, \
        label_placeholder, save_file):
        """
        Run the network
        """
        # Launch the graph
        # Initializing the variables
        init = tf.initialize_all_variables()

        saver = tf.train.Saver()

        time0 = time()

        with tf.Session() as sess:
            sess.run(init)

            # run_id = log.dl_run_start(dl_run, dl_network, dl_model_file_path, dl_data, hyper_dict)

            # Training cycle
            for epoch in range(training_epochs):
                time1 = time()
                total_batch = int(data_size / batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = next_batch(batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    sess.run(self.optimizer, feed_dict={input_placeholder: batch_x, \
                        label_placeholder: batch_y})
                # Display logs per epoch step
                cost = sess.run(self.cost, feed_dict={input_placeholder: batch_x, \
                    label_placeholder: batch_y})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(cost))
                print("Time elapsed:", "{:.2f}".format(time() - time1))
            saver.save(sess, save_file)
            print("Training Finished!")
            print("Total training time:", "{:.2f}".format(time() - time0))
            return saver

    def evaluate(self, saver, test_inputs, test_labels, input_placeholder, label_placeholder):
        """
        Evaluate accuracy with test data.
        """
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('.'))

            # Test model
            correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(test_labels, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            test_accuracy  = sess.run(accuracy, feed_dict={input_placeholder: test_inputs, \
                label_placeholder: test_labels})

            print("Accuracy:", test_accuracy)
