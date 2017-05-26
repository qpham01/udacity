""" Sentiment analysis in TensorFlow """
import numpy as np
from string import punctuation
from sklearn.model_selection import train_test_split
import tensorflow as tf

REVIEW_FILE = '../sentiment_network/reviews.txt'
LABEL_FILE = '../sentiment_network/labels.txt'

def prepare_reviews(reviews, labels):
    """ Prepare reviews for sentiment analysis """
    # remove punctuation from reviews
    all_text = ''.join([c for c in reviews if c not in punctuation])

    # remove new lines from reviews
    reviews = all_text.split('\n')

    all_text = ' '.join(reviews)
    words = all_text.split()

    all_labels = labels.split('\n')

    labels = [1 if label == 'positive' else 0 for label in all_labels]

    return words, reviews, labels

def make_integer_vocabulary(text):
    """ Encode words in  as integers """
    vocab = set(text)
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    return vocab_to_int, int_to_vocab

with open(REVIEW_FILE, 'r') as fread:
    REVIEWS = fread.read()
with open(LABEL_FILE, 'r') as fread:
    LABEL_TEXT = fread.read()

WORDS, REVIEWS, LABELS = prepare_reviews(REVIEWS, LABEL_TEXT)
print(LABELS[0:20])

VOCAB_TO_INT, INT_TO_VOCAB = make_integer_vocabulary(WORDS)

def encode_reviews(reviews, vocab_to_int):
    """ Convert the reviews to integers, same shape as reviews list, but with integers """
    reviews_ints = []
    for review in reviews:
        review_words = review.split()
        int_review = np.array([vocab_to_int[word] for word in review_words], dtype=np.int32)
        reviews_ints.append(int_review)

    return reviews_ints

INT_REVIEWS = encode_reviews(REVIEWS, VOCAB_TO_INT)

print(len(INT_REVIEWS))
INT_REVIEWS = [x for x in INT_REVIEWS if len(x) > 0]
print(len(INT_REVIEWS))

def prepare_review_input(max_length, int_reviews):
    """
    Prepare each review for input, but with a specified maximum length
    Pad each review with 0 from the front.
    """
    shape = (len(int_reviews), max_length)
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

def make_network_graph(lstm_size, lstm_layers, batch_size, vector_size, embed_size, learning_rate):
    """ Create the recurrent neural network graph """
    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():
        inputs_ = tf.placeholder(tf.float32, [batch_size, vector_size], name='inputs')
        labels_ = tf.placeholder(tf.int32, batch_size, name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        embedding = tf.Variable(dtype=)
        embed = 