import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float('inf')
        best_count = 0
        logN = math.log(len(self.X))
        for p in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(p)
                logL = model.score(self.X, lengths=self.lengths)
                score = -2 * logL + p * logN
            except:
                continue
            if score < best_score:
                best_score = score
                best_count = p

        return self.base_model(best_count)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    all_scores = None


    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str, \
        n_constant=3, min_n_components=2, max_n_components=10, random_state=14, verbose=False):
        super(SelectorDIC, self).__init__(all_word_sequences, all_word_Xlengths, this_word, \
            n_constant=n_constant, min_n_components=min_n_components, \
            max_n_components=max_n_components, random_state=random_state, verbose=verbose)
        # A dictionary to store all scores
        if SelectorDIC.all_scores is None:
            SelectorDIC.all_scores = dict()
            print("Making database of scores for all words and their model state counts.")
            for num_states in range(min_n_components, max_n_components + 1):
                print("Creating scores for state count {}".format(num_states))
                SelectorDIC.all_scores[num_states] = dict()
                for word in self.words:
                    X, lengths = self.hwords[word]
                    try:
                        model = self.base_model(num_states)
                        score = model.score(X, lengths=lengths)
                    except:
                        score = 0 # ignore models that don't work

                    SelectorDIC.all_scores[num_states][word] = score

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Need to compute the scores of all words with all components for comparison

        best_score = float('-inf') # best DIC score is highest score
        best_count = 0
        m_prime = len(self.words) - 1 # m_prime = M - 1

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            scores = []
            for word in self.words:
                if word == self.this_word:
                    continue
                scores.append(SelectorDIC.all_scores[num_states][word])

            score = SelectorDIC.all_scores[num_states][self.this_word] - sum(scores) / m_prime

            if score > best_score:
                best_score = score
                best_count = num_states

        return self.base_model(best_count)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    recognize = False

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        global recognize

        if not SelectorCV.recognize:
            # Handle KFold cross validation
            split_method = KFold(n_splits=2)
            word_sequences = self.words[self.this_word]
            train_list = []
            test_list = []
            for train_idx, test_idx in split_method.split(word_sequences):
                train_list.extend(train_idx)
                test_list.extend(test_idx)
            self.X, self.lengths = combine_sequences(train_list, word_sequences)
            test_X, test_lengths = combine_sequences(test_list, word_sequences)

        # TODO implement model selection using CV
        best_score = float('-inf')
        best_count = 0
        logN = math.log(len(self.X))
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_states)
                if SelectorCV.recognize:
                    score = model.score(self.X, lengths=self.lengths)
                else:
                    score = model.score(test_X, lengths=test_lengths)
            except:
                continue
            if score > best_score:
                best_score = score
                best_count = num_states

        return self.base_model(best_count)
