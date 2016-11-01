'''
Calculate later words probabilities
'''
import operator
from next_word import next_word_probability
from next_word import probability_from_count
from next_word import SAMPLE_MEMO

#------------------------------------------------------------------

#
#   Bayes Optimal Classifier
#
#   In this quiz we will compute the optimal label for a second missing word in a row
#   based on the possible words that could be in the first blank
#
#   Finish the procedurce, LaterWords(), below
#
#   You may want to import your code from the previous programming exercise!
#

CORRUPTED_MEMO = '''
Yeah, I'm gonna --- you to go ahead --- --- complain about this. Oh, and if you could --- --- and sit at the kids' table, that'd be --- 
'''

DATA_LIST = SAMPLE_MEMO.strip().split()

WORDS_TO_GUESS = ["ahead", "could"]

def later_words_probabilities(sample, word, distance):
    '''
    @param sample: a sample of text to draw from
    @param word: a word occuring before a corrupted sequence
    @param distance: how many words later to estimate (i.e. 1 for the next word, 2 for the word
        after that)
    @returns: a single word which is the most likely possibility
    '''
    # Given a word, collect the relative probabilities of possible following words
    # from @sample. You may want to import your code from the maximum likelihood exercise.
    next_words = next_word_probability(sample, word)
    next_words = probability_from_count(next_words)
    node = []
    node.append(next_words)
    # Repeat the above process--for each distance beyond 1, evaluate the words that
    # might come after each word, and combine them weighting by relative probability
    # into an estimate of what might appear next.
    if distance > 1:
        children = dict()
        for a_word in next_words.iterkeys():
            children[a_word] = later_words_probabilities(sample, a_word, distance - 1)
        node.append(children)

    return node

def leaf_probabilities(root):
    '''
    Calculates the probabilities of the leaf nodes given the root of a probability tree
    '''
    leaf_prob = dict()
    return leaf_prob

def best_later_word(sample, word, distance):
    '''
    Calculate the word with the maximum probabilities some distance after given word
    '''
    prob_tree = later_words_probabilities(sample, word, distance)
    leaf_prob = leaf_probabilities(prob_tree)

    best_word = max(leaf_prob.iteritems(), key=operator.itemgetter(1))[0]
    best_word_prob = leaf_prob[best_word]
    return best_word_prob

#print best_later_word(SAMPLE_MEMO, "ahead", 2)
TEST_TEXT2 = 'This is a test. That is not a test and is a mess.'
print later_words_probabilities(TEST_TEXT2, 'is', 2)
