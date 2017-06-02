""" Sentiment analysis in TensorFlow """
from collections import Counter
from string import punctuation
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

class SentimentAnalyzer:
    """ Sentiment analysis """
    def __init__(self):
        # Text data members
        self.words = None
        self.raw_features = None
        self.raw_labels = None
        self.features = None
        self.feature_size = 0
        self.embeddings = None
        self.vocab_to_int = None
        self.int_to_vocab = None
        self.int_words = None
        self.sorted_vocab = None
        self.vocab_size = 0
        self.embedding_size = 0
        self.int_features = None

        # Train, validation, test
        self.train_x = None
        self.train_y = None
        self.valid_x = None
        self.valid_y = None
        self.test_x = None
        self.test_y = None

        # TensorFlow graph data members
        self.graph = None
        self.inputs = None
        self.labels = None
        self.keep_prob = None
        self.embedding = None
        self.embed = None
        self.lstm = None
        self.drop = None
        self.cell = None
        self.initial_state = None
        self.softmax_w = None
        self.softmax_b = None
        self.outputs = None
        self.predictions = None
        self.final_state = None
        self.loss = None
        self.cost = None
        self.optimizer = None
        self.accuracy = None
        self.correct_pred = None

        # Hyper parameters
        self.lstm_size = 0
        self.lstm_layers = 0
        self.batch_size = 0
        self.embed_size = 0
        self.learning_rate = 0
        self.epochs = 0

    def prepare_training_data(self, raw_features, raw_labels, train_fraction=0.8, \
        validate_fraction=0.1, feature_size=200):
        """ Prepare data for training """
        self.feature_size = feature_size
        self._prepare_data(raw_features, raw_labels)
        self._create_vocabulary()
        self._encode_features()
        self._prepare_features()
        self.train_x, self.valid_x, self.test_x = \
            self._train_test_validation_split(self.features, train_fraction, validate_fraction)
        self.train_y, self.valid_y, self.test_y = \
            self._train_test_validation_split(self.labels, train_fraction, validate_fraction)

    def _prepare_data(self, raw_features, raw_labels):
        """ Prepare reviews for sentiment analysis """
        # remove punctuation from reviews
        all_text = ''.join([c for c in raw_features if c not in punctuation])

        # remove new lines from reviews
        self.raw_features = all_text.split('\n')

        self.words = ' '.join(self.raw_features).split()

        all_labels = raw_labels.split('\n')

        self.labels = [1 if label == 'positive' else 0 for label in all_labels]

    def _create_vocabulary(self):
        """
        Create the vocabulary from the text corpus used for training.

        Parameters:
        """
        word_counts = Counter(self.words)
        self.sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        self.int_to_vocab = {idx: word for idx, word in enumerate(self.sorted_vocab)}
        self.vocab_to_int = {word: idx for idx, word in enumerate(self.sorted_vocab, 1)}

    def _encode_features(self):
        """ Convert the reviews to integers, same shape as reviews list, but with integers """
        int_features = []
        for feature in self.raw_features:
            feature_words = feature.split()
            int_feature = np.array([self.vocab_to_int[word] for word in feature_words], \
                dtype=np.int32)
            int_features.append(int_feature)

        # Remove any empty feature
        non_zero_idx = [ii for ii, feature in enumerate(int_features) if len(feature) != 0]
        self.int_features = [int_features[ii] for ii in non_zero_idx]
        self.labels = np.array([self.labels[ii] for ii in non_zero_idx])

    def _prepare_features(self):
        """
        Prepare each review for input, but with a specified maximum length
        Pad each review with 0 from the front.
        """
        shape = (len(self.int_features), self.feature_size)
        self.features = np.zeros(shape, dtype=np.int32)
        for i, feature in enumerate(self.int_features):
            start = max(0, self.feature_size - len(feature))
            self.features[i, start:self.feature_size] = feature[:min(self.feature_size, \
                len(feature))]
        #print("Feature sample", self.features[:10,:100])

    def _train_test_validation_split(self, data, train_fraction, validation_fraction):
        """ Split data into train, validation, and test sets per specified fractions. """
        '''
        test_val_fraction = 1.0 - train_fraction
        train, val = train_test_split(data, train_size=train_fraction)
        val, test = train_test_split(val, train_size=validation_fraction / test_val_fraction)
        '''
        split_idx = int(len(data) * train_fraction)
        train, valid = data[:split_idx], data[split_idx:]

        test_idx = int(len(valid) * 0.5)
        valid, test = valid[:test_idx], valid[test_idx:]

        return train, valid, test

    def make_training_graph(self, lstm_size, lstm_layers, batch_size, embed_size, learning_rate):
        """ Create the recurrent neural network graph """
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.learning_rate = learning_rate

        # Create the graph object
        self.graph = tf.Graph()
        # Add nodes to the graph
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
            self.graph_labels = tf.placeholder(tf.int32, [None, None], name='labels')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            self.embedding = tf.Variable(tf.random_uniform((self.vocab_size, embed_size), -1, 1))
            self.embed = tf.nn.embedding_lookup(self.embedding, self.inputs)

            # Your basic LSTM cell
            self.lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

            # Add dropout to the cell
            self.drop = tf.contrib.rnn.DropoutWrapper(self.lstm, output_keep_prob=self.keep_prob)

            # Stack up multiple LSTM layers, for deep learning
            self.cell = tf.contrib.rnn.MultiRNNCell([self.drop] * lstm_layers)

            # Getting an initial state of all zeros
            self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

            self.outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, self.embed, \
                initial_state=self.initial_state)
            self.predictions = tf.contrib.layers.fully_connected(self.outputs[:, -1], 1, \
                activation_fn=tf.sigmoid)
            self.cost = tf.losses.mean_squared_error(self.graph_labels, self.predictions)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            self.correct_pred = tf.equal(tf.cast(tf.round(self.predictions), tf.int32), \
                self.graph_labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def get_batches(self, features, labels, batch_size):
        """ Get a batch for training """
        n_batches = len(features) // batch_size
        features, labels = features[:n_batches * batch_size], labels[:n_batches * batch_size]
        for idx in range(0, len(features), batch_size):
            yield features[idx : idx + batch_size], labels[idx : idx + batch_size]

    def train(self, epochs):
        """ Train on sentiment analysis """
        self.epochs = epochs
        with self.graph.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1
            for epoch in range(epochs):
                state = sess.run(self.initial_state)

                for (features, labels) in self.get_batches(self.train_x, self.train_y, \
                    self.batch_size):
                    feed = {self.inputs: features,
                            self.graph_labels: labels[:, None],
                            self.keep_prob: 0.5,
                            self.initial_state: state}
                    loss, state, _ = sess.run([self.cost, self.final_state, self.optimizer], \
                        feed_dict=feed)

                    if iteration % 5 == 0:
                        print("Epoch: {}/{}".format(epoch + 1, epochs), \
                            "Iteration: {}".format(iteration), \
                            "Train loss: {:.3f}".format(loss))

                    if iteration % 25 == 0:
                        val_acc = []
                        val_state = sess.run(self.cell.zero_state(self.batch_size, tf.float32))
                        for features, labels in self.get_batches(self.valid_x, self.valid_y, \
                            self.batch_size):
                            feed = {self.inputs: features,
                                    self.graph_labels: labels[:, None],
                                    self.keep_prob: 1,
                                    self.initial_state: val_state}
                            batch_acc, val_state = sess.run([self.accuracy, self.final_state], \
                                feed_dict=feed)
                            val_acc.append(batch_acc)
                        print("Val acc: {:.3f}".format(np.mean(val_acc)))
                    iteration += 1
            saver.save(sess, "checkpoints/sentiment.ckpt")

    def test(self):
        """ Test the sentiment analysis """
        test_acc = []
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            test_state = sess.run(self.cell.zero_state(self.batch_size, tf.float32))
            for features, labels in self.get_batches(self.test_x, self.test_y, self.batch_size):
                feed = {self.inputs: features,
                        self.graph_labels: labels[:, None],
                        self.keep_prob: 1,
                        self.initial_state: test_state}
                batch_acc, test_state = sess.run([self.accuracy, self.final_state], feed_dict=feed)
                test_acc.append(batch_acc)
            return test_acc
