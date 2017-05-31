""" Process movie review data for sentiment analysis """

from string import punctuation

REVIEW_FILE = '../sentiment_network/reviews.txt'
LABEL_FILE = '../sentiment_network/labels.txt'

def prepare_reviews(reviews, labels):
    """ Prepare reviews for sentiment analysis """
    # remove punctuation from reviews
    all_text = ''.join([c for c in reviews if c not in punctuation])

    # remove new lines from reviews
    reviews = all_text.split('\n')

    all_text = ' '.join(reviews)
    words = all_text.split()

    all_labels = labels.split('\n')

    labels = [1 if label == 'positive' else 0 for label in all_labels]

    return words, reviews, labels

def make_integer_vocabulary(text):
    """ Encode words in  as integers """
    vocab = set(text)
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    return vocab_to_int, int_to_vocab, vocab


with open(REVIEW_FILE, 'r') as fread:
    REVIEWS = fread.read()
with open(LABEL_FILE, 'r') as fread:
    LABEL_TEXT = fread.read()

WORDS, REVIEWS, LABELS = prepare_reviews(REVIEWS, LABEL_TEXT)
print(LABELS[0:20])

