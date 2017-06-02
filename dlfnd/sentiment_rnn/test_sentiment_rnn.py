""" Tests for sentiment analysis """
import pickle
import unittest
import numpy as np
from sentiment_rnn import SentimentAnalyzer
from data_reviews import prepare_review_data

class TestSentiment(unittest.TestCase):
    """ Contains tests for sentiment analysis """
    reviews = None
    labels = None
    feature_size = 200
    def __init__(self, *args, **kwargs):
        super(TestSentiment, self).__init__(*args, **kwargs)
        if TestSentiment.reviews is None:
            TestSentiment.reviews, TestSentiment.labels = prepare_review_data()
        self.analyzer = SentimentAnalyzer()
        self.analyzer.prepare_training_data(TestSentiment.reviews, TestSentiment.labels)

        lstm_size = 256
        lstm_layers = 1
        batch_size = 500
        embed_size = 300
        learning_rate = 0.001

        self.analyzer.make_training_graph(lstm_size, lstm_layers, batch_size, embed_size, \
            learning_rate)

    def test_01_prepare_training_data(self):
        """ Test review preparation """
        print(self.analyzer.train_x[0, 190:200])
        print(self.analyzer.train_y[0:20])
        print(self.analyzer.train_x.shape)
        print(self.analyzer.sorted_vocab[0:10])
        assert len(self.analyzer.words) > 6000000
        assert self.analyzer.train_x.shape == (20000, 200)
        assert self.analyzer.valid_x.shape == (2500, 200)
        assert self.analyzer.test_x.shape == (2500, 200)
        data = {}
        data['train_x'] = self.analyzer.train_x
        data['train_y'] = self.analyzer.train_y
        data['valid_x'] = self.analyzer.valid_x
        data['valid_y'] = self.analyzer.valid_y

        with open('test_data.p', 'wb') as fwrite:
            pickle.dump(data, fwrite)

    def test_02_make_training_graph(self):
        """ Test making of the training graph """
        output_shape = self.analyzer.outputs.get_shape()
        assert output_shape[0] == self.analyzer.batch_size and \
            output_shape[2] == self.analyzer.lstm_size

    def test_03_training(self):
        """ Test training on sentiment analysis """
        epochs = 10
        self.analyzer.train(epochs)

    def test_04_testing(self):
        """ Test inference on test set """
        test_acc = np.mean(self.analyzer.test())
        print("Test accuracy: {:.3f}".format(test_acc))
        assert test_acc > 0.6

if __name__ == '__main__':
    unittest.main()
