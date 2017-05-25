"""
Create word2vec embeddings from a text corpus
"""
import random
import numpy as np
from collections import Counter
from utils import preprocess, create_lookup_tables

class Word2Vec:
    """ Encapsulates word2vec embeddings """
    def __init__(self, text):
        self.text = text
        self.words = preprocess(self.text)
        self.embeddings = None
        self.vocab_to_int, self.int_to_vocab = create_lookup_tables(self.words)
        self.int_words = [self.vocab_to_int[self.word] for self.word in self.words]
        self.drop_count = 0
        self.total_count = 0

    def print_data_sample(self, start=0, end=20):
        """ print out a range of sample words """
        print("words from {} to {}".format(start, end))
        print(self.words[start:end])

    def print_data_stats(self):
        """ prints out some standard statistics on Word2Vec """
        print("Total words: {}".format(len(self.words)))
        print("Unique words: {}".format(len(set(self.words))))

    def subsample(self, threshold=1e-3):
        """ subsample the text corpus to remove infrequent words """
        word_counts = Counter(self.int_words)
        self.total_count = len(self.int_words)
        freqs = {word: count / self.total_count for word, count in word_counts.items()}
        p_drop = {word: 1 - np.sqrt(threshold / freqs[word]) for word in word_counts}
        train_words = [word for word in self.int_words if random.random() > p_drop[word]]
        self.drop_count = self.total_count - len(train_words)
        print('Subsampling dropped {} words out of {} words ({:.2f}%)'.format(drop_count, total_count, 100 * drop_count / total_count))
