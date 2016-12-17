'''
Unit tests for maximum_likelihood module
'''
import unittest
from next_word import next_word_probability
from next_word import probability_from_count
from later_words import later_words_probabilities
from later_words import best_later_word

TEST_TEXT1 = 'This is a test. That is not a test and is a mess.'
ONE_THIRD = 1.0/3.0
TWO_THIRDS = 2.0/3.0

class TestProbability(unittest.TestCase):
    '''Test next_word_probability'''
    def test_next_word_is(self):
        '''Test next_word_probability for is'''
        next_words = next_word_probability(TEST_TEXT1, 'is')
        self.assertEqual(2, len(next_words), 2)
        self.assertEqual(2, next_words['a'], 2)
        self.assertEqual(1, next_words['not'], 1)
        next_words = probability_from_count(next_words)
        self.assertEqual(TWO_THIRDS, next_words['a'])
        self.assertEqual(ONE_THIRD, next_words['not'])

    def test_next_word_a(self):
        '''Test next_word_probability for a'''
        next_words = next_word_probability(TEST_TEXT1, 'a')
        self.assertEqual(2, len(next_words))
        self.assertEqual(2, next_words['test'])
        self.assertEqual(1, next_words['mess'])
        next_words = probability_from_count(next_words)
        self.assertEqual(TWO_THIRDS, next_words['test'])
        self.assertEqual(ONE_THIRD, next_words['mess'])

    def test_later_words_prob_a(self):
        '''Test later_word_probability for a'''
        later_words = later_words_probabilities(TEST_TEXT1, 'a', 1)
        self.assertEqual(1, len(later_words))
        self.assertEqual(TWO_THIRDS, later_words[0]['test'])
        self.assertEqual(ONE_THIRD, later_words[0]['mess'])

    def test_later_words_prob_is(self):
        '''Test later_word_probability for is'''
        later_words = later_words_probabilities(TEST_TEXT1, 'is', 2)
        self.assertEqual(2, len(later_words))
        self.assertEqual(TWO_THIRDS, later_words[0]['a'])
        self.assertEqual(ONE_THIRD, later_words[0]['not'])
        self.assertEqual(TWO_THIRDS, later_words[1]['a'][0]['test'])
        self.assertEqual(ONE_THIRD, later_words[1]['a'][0]['mess'])
        self.assertEqual(1, later_words[1]['not'][0]['a'])

    def test_best_later_words(self):
        '''Test best_later_word'''
        best = best_later_word(TEST_TEXT1, 'is', 2)
        print best
        self.assertEqual('test', best[0])
        self.assertAlmostEqual(0.44444444, best[1])

if __name__ == '__main__':
    unittest.main()
