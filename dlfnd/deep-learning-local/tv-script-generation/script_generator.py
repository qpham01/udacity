""" Generate TV scripts """
import tensorflow as tf

class ScriptMaker:
    """
    A class for generating TV scripts
    """
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

class ScriptMaker(TfTextRnn):
    """ A class for making scripts """
    def __init__(self):
        TfTextRnn.__init__(self)
        
    def create_vocabulary(self, text):
        """
        Create lookup tables for vocabulary
        :param text: The text of tv scripts split into words
        :return: A tuple of dicts (vocab_to_int, int_to_vocab)
        """
        self.words = text
        word_counts = Counter(self.words)
        self.sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        self.int_to_vocab = {idx: word for idx, word in enumerate(self.sorted_vocab)}
        self.vocab_to_int = {word: idx for idx, word in enumerate(self.sorted_vocab)}
        self.vocab_size = len(self.vocab_to_int)    

    def create_punctuation_lookup(self):
        """
        Generate a dict to turn punctuation into a token.
        :return: Tokenize dictionary where the key is the punctuation and the value is the token
        """
        punctuations = ['.', ',', '"', ';', '!', '?', '(', ')', '--', '\n']
        tokens = ['<PERIOD>', '<COMMA>', '<DQUOTE>', '<SEMICOLON>', '<BANG>', '<QUESTION>', '<PAREN1>', '<PAREN2>', '<DDASH>', '<NEWLINE>']
        self.punctuation_lookup = {x: y for (x,y) in zip(punctuations, tokens)}
        
    def make_training_graph(self, rnn_size, rnn_layers, embed_dim, inputs=None):
        """
        Generates the training graph for making scripts
        """
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.embed_dim = embed_dim
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            # define inputs
            if inputs is None:
                self.inputs = tf.placeholder(tf.int32, [None, None], name='input')
            else:
                self.inputs = inputs
            self.targets = tf.placeholder(tf.int32, [None, None], name='target')
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
            # sets up RNN cell
            self.lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)

            # Add dropout to the cell
            self.drop = tf.contrib.rnn.DropoutWrapper(self.lstm, output_keep_prob=self.keep_prob)

            # Stack up multiple LSTM layers, for deep learning
            self.cell = tf.contrib.rnn.MultiRNNCell([self.drop] * self.rnn_layers)

            # Getting an initial state of all zeros
            self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
            self.initial_state = tf.identity(self.initial_state, name='initial_state')
            
            # Embeddings
            self.embedding = tf.Variable(tf.random_uniform((self.vocab_size, self.embed_dim), -1, 1))
            self.embed = tf.nn.embedding_lookup(self.embedding, self.inputs)
            
            # Outputs and final state
            self.outputs, final_state = tf.nn.dynamic_rnn(self.cell, self.embed, dtype=tf.float32)
            self.final_state = tf.identity(final_state, name='final_state')
            self.logits = tf.contrib.layers.fully_connected(self.outputs, self.vocab_size, activation_fn=None)