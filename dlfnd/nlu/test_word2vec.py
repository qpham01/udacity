""" Unit tests for Word2Vec """
import unittest

from data_text8 import get_text8
from word2vec import Word2Vec

class TestWord2Vec(unittest.TestCase):
    """ Test class for Word2Vec """
    def __init__(self, *args, **kwargs):
        super(TestWord2Vec, self).__init__(*args, **kwargs)
        self.word2vec = Word2Vec()
    
    def test_01_preparing_data(self):
        """ Test preparing data for word2vec """
        text = get_text8()
        self.word2vec.prepare_text(text)
        self.word2vec.print_data_stats()
        self.word2vec.print_top_dropped_words()
        train_fraction = len(self.word2vec.train_words) / self.word2vec.total_count
        assert train_fraction < 0.7

    def test_02_training(self):
        """ Test the training of word embeddings """
        text = get_text8()
        self.word2vec.prepare_text(text)
        self.word2vec.make_embedding_graph()

        epochs = 1
        batch_size = 10000
        window_size = 10

        self.word2vec.train_embedding('checkpoints', 'text8.ckpt', epochs, batch_size, \
            window_size=window_size)
    
    def test_03_load_embeddings(self):
        """ Test loading of the embedding matrix """
        text = get_text8()
        self.word2vec.prepare_text(text)
        self.word2vec.make_embedding_graph()

        embed_matrix = self.word2vec.load_embeddings('checkpoints')
        shape = embed_matrix.shape
        assert shape == (self.word2vec.vocab_size, self.word2vec.embedding_size)

if __name__ == '__main__':
    unittest.main()
