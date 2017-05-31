""" Unit tests for Word2Vec """
import unittest
import pickle
from data_text8 import get_text8
from word2vec import Word2Vec

class TestWord2Vec(unittest.TestCase):
    """ Test class for Word2Vec """
    def __init__(self, *args, **kwargs):
        super(TestWord2Vec, self).__init__(*args, **kwargs)
        self.word2vec = Word2Vec()

    def test_01_preparing_data(self):
        """ Test preparing data for word2vec """
        self.word2vec.load_text8()

        train_fraction = len(self.word2vec.train_words) / self.word2vec.total_count
        assert train_fraction < 0.7

    def test_02_training(self):
        """ Test the training of word embeddings """
        self.word2vec.load_text8()

        epochs = 1
        batch_size = 10000
        window_size = 10

        self.word2vec.train_embedding('checkpoints', 'text8.ckpt', epochs, batch_size, \
            window_size=window_size)

    def test_03_load_embeddings(self):
        """ Test loading of the embedding matrix """
        self.word2vec.load_text8()

        embed_matrix = self.word2vec.load_embeddings('checkpoints')
        shape = embed_matrix.shape
        assert shape == (self.word2vec.vocab_size, self.word2vec.embedding_size)

    def test_04_pickle_embeddings(self):
        """ Test pickling of embeddings """
        save = "checkpoints"
        name = "text8"
        file_path = "data/text8_embeddings.p"

        self.word2vec.load_text8()

        self.word2vec.pickle_embeddings(save, file_path)

        with open(file_path, 'rb') as fread:
            restore = pickle.load(fread)

        assert "embeddings" in restore
        embed_matrix = restore["embeddings"]
        shape = embed_matrix.shape
        assert shape == (self.word2vec.vocab_size, self.word2vec.embedding_size)
    
if __name__ == '__main__':
    unittest.main()
