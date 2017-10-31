import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # Implement the recognizer
    # return probabilities, guesses
    sequences = test_set.get_all_sequences()
    x_lengths = test_set.get_all_Xlengths()
    num_test_words = len(sequences)
    for index in range(num_test_words):
        prob_dict = {}
        best_score = 0
        best_guess = 'NONE'
        for word, model in models.items():
            X, lengths = x_lengths[index]
            if model is None:
                score = 0
            else:
                try:
                    score = model.score(X, lengths=lengths)
                except ValueError:
                    score = 0
            prob_dict[word] = score
            if score > best_score:
                best_score = score
                best_guess = word
        guesses.append(best_guess)
        probabilities.append(prob_dict)

    return probabilities, guesses
