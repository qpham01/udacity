""" A library of classes for working with TensorFlow """

from collections import Counter

class TfTextRnn:
    """ A base class for working with RNN for text processing in TensorFlow """
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
        self.punctuation = None
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

    def _create_vocabulary(self):
        """
        Create the vocabulary from the text corpus used for training.

        Parameters:
        """
        word_counts = Counter(self.words)
        self.sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        self.int_to_vocab = {idx: word for idx, word in enumerate(self.sorted_vocab)}
        self.vocab_to_int = {word: idx for idx, word in enumerate(self.sorted_vocab, 1)}
        self.vocab_size = len(self.vocab_to_int)
    