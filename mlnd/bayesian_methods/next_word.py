'''
maximum_likelihood:  demonstrates maximum likelihood using bayesian methods
'''
import re

SAMPLE_MEMO = '''Milt, we're gonna need to go ahead and move you downstairs into storage B.
We have some new people coming in, and we need all the space we can get. So if you could just
go ahead and pack up your stuff and move it down there, that would be terrific, OK?
Oh, and remember: next Friday... is Hawaiian shirt day. So, you know, if you want to, go
ahead and wear a Hawaiian shirt and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come in on Sunday, too...
Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead and come in tomorrow. So if
you could be here around 9 that would be great, mmmk... oh oh! and I almost forgot ahh, I'm also
gonna need you to go ahead and come in on Sunday too, kay. We ahh lost some people this week and
ah, we sorta need to play catch up.
'''

#
#   Maximum Likelihood Hypothesis
#
#
#   In this quiz we will find the maximum likelihood word based on the preceding word
#
#   Fill in the NextWordProbability procedure so that it takes in sample text and a word,
#   and returns a dictionary with keys the set of words that come after, whose values are
#   the number of times the key comes after that word.
#
#   Just use .split() to split the sample_memo text into words separated by spaces.

def next_word_probability(sampletext, word):
    '''Returns a dictionary of words that comes after word in sampletext'''
    separators = r'\s|, |\.\.\. |\. |\.|\! |\?|\n'
    all_words = re.split(separators, sampletext)
    word_dict = dict()
    prev_word = ''
    for next_word in all_words:
        if prev_word != word:
            prev_word = next_word
            continue
        if word_dict.has_key(next_word):
            word_dict[next_word] += 1
        else:
            word_dict[next_word] = 1
        prev_word = next_word
    return word_dict

def probability_from_count(word_dict):
    '''Returns a dictionary of words and probabilities from a dictionary of words and counts'''
    # calculate probabilities
    total_count = float(sum(word_dict.itervalues()))
    for key, value in word_dict.iteritems():
        word_dict[key] = float(value) / total_count
    return word_dict

#RESULT = next_word_probability('this, is! a... test. ', 'is')
#print RESULT
#print len(RESULT)
