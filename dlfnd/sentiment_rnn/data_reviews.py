""" Movie review data """

REVIEW_FILE = '../sentiment_network/reviews.txt'
LABEL_FILE = '../sentiment_network/labels.txt'

def prepare_review_data():
    """ Read in review data file and prepare data once for testing """
    with open(REVIEW_FILE, 'r') as fread:
        reviews = fread.read()
    with open(LABEL_FILE, 'r') as fread:
        labels = fread.read()
    return reviews, labels
