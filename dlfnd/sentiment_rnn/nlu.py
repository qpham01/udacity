""" Contains code for natural language understanding """
from collections import Counter
import numpy as np
from tqdm import tqdm
import tensorflow as tf

class TextProcessor:
    """ Base class for processing text """
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


