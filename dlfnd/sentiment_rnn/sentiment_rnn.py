""" Sentiment analysis in TensorFlow """
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

class SentimentAnalyzer:
    """ Sentiment analysis """
    def __init__(self):
        # Text data members
        self.words = None
        self.embeddings = None
        self.vocab_to_int = None
        self.int_to_vocab = None
        self.int_words = None
        self.sorted_vocab = None
        self.vocab_size = 0
        self.embedding_size = 0
        self.reviews_ints = None

        # TensorFlow graph data members
        self.graph = None
        self.inputs = None
        self.labels = None
        self.embedding = None
        self.embed = None
        self.softmax_w = None
        self.softmax_b = None
        self.loss = None
        self.cost = None
        self.optimizer = None

    def create_vocabulary(self, text):
        """
        Create the vocabulary from the text corpus used for training.

        Parameters:
        text: text corpus ready for tokenization
        labels: the corresponding labels
        """
        word_counts = Counter(text)
        self.sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        self.int_to_vocab = {idx: word for idx, word in enumerate(self.sorted_vocab)}
        self.vocab_to_int = {word: idx for idx, word in self.int_to_vocab.items()}

    def encode_reviews(self, reviews):
        """ Convert the reviews to integers, same shape as reviews list, but with integers """
        self.reviews_ints = []
        for review in reviews:
            review_words = review.split()
            int_review = np.array([self.vocab_to_int[word] for word in review_words], \
                dtype=np.int32)
            self.reviews_ints.append(int_review)

INT_REVIEWS = encode_reviews(REVIEWS, VOCAB_TO_INT)

print(len(INT_REVIEWS))
INT_REVIEWS = [x for x in INT_REVIEWS if len(x) > 0]
print(len(INT_REVIEWS))

    def prepare_review_input(self, max_length):
    """
    Prepare each review for input, but with a specified maximum length
    Pad each review with 0 from the front.
    """
    shape = (len(self.int_reviews), max_length)
    input_reviews = np.zeros(shape, dtype=np.int32)
    for i, review in enumerate(int_reviews):
        start = max(0, max_length - len(review))
        input_reviews[i, start:max_length] = review[:min(max_length, len(review))]
    return input_reviews

REVIEW_LENGTH = 200
FEATURES = prepare_review_input(REVIEW_LENGTH, INT_REVIEWS)
print(FEATURES[:2, :100])

def train_test_validation_split(data, train_fraction, validation_fraction):
    """ Split data into train, validation, and test sets per specified fractions. """
    test_val_fraction = 1.0 - train_fraction
    train, val = train_test_split(data, train_size=train_fraction)
    val, test = train_test_split(val, train_size=validation_fraction / test_val_fraction)
    return train, val, test

TRAIN_X, VALID_X, TEST_X = train_test_validation_split(FEATURES, 0.8, 0.1)
TRAIN_Y, VALID_Y, TEST_Y = train_test_validation_split(LABELS, 0.8, 0.1)

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(TRAIN_X.shape),
      "\nValidation set: \t{}".format(VALID_X.shape),
      "\nTest set: \t\t{}".format(TEST_X.shape))

LSTM_SIZE = 256
LSTM_LAYERS = 1
BATCH_SIZE = 500
LEARNING_RATE = 0.001
EMBED_SIZE = 100

def make_network_graph(lstm_size, lstm_layers, batch_size, vector_size, embed_size, learning_rate):
    """ Create the recurrent neural network graph """
    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():
        inputs_ = tf.placeholder(tf.float32, [batch_size, vector_size], name='inputs')
        labels_ = tf.placeholder(tf.int32, batch_size, name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        embedding = tf.Variable(tf.random_uniform((N_WORDS, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs_)

        # Your basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)

        # Getting an initial state of all zeros
        initial_state = cell.zero_state(batch_size, tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
        predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
        cost = tf.losses.mean_squared_error(labels_, predictions)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        return graph

GRAPH = make_network_graph(LSTM_SIZE, LSTM_LAYERS, BATCH_SIZE, REVIEW_LENGTH, EMBED_SIZE, LEARNING_RATE)

def get_batches(x, y, batch_size=100):
    """ Get a batch for training """
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches * batch_size]
    for idx in range(0, len(x), batch_size):
        yield x[idx : idx + batch_size], y[idx : idx + batch_size]

EPOCHS = 10

def train(graph, epochs):
    """ Train on sentiment analysis """
    with graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 1
        for e in range(epochs):
            state = sess.run(initial_state)
            
            for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob: 0.5,
                        initial_state: state}
                loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
                
                if iteration%5==0:
                    print("Epoch: {}/{}".format(e, epochs),
                        "Iteration: {}".format(iteration),
                        "Train loss: {:.3f}".format(loss))

                if iteration%25==0:
                    val_acc = []
                    val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                    for x, y in get_batches(val_x, val_y, batch_size):
                        feed = {inputs_: x,
                                labels_: y[:, None],
                                keep_prob: 1,
                                initial_state: val_state}
                        batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                        val_acc.append(batch_acc)
                    print("Val acc: {:.3f}".format(np.mean(val_acc)))
                iteration +=1
        saver.save(sess, "checkpoints/sentiment.ckpt")

train(GRAPH, EPOCHS, )