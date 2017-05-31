""" Test natural language understanding """
import os
import unittest
from named_entity import NamedEntityExtractor
from data_glove import get_glove_vectors

class TestNlu(unittest.TestCase):
    """ Test natural language understanding """
    entities = {\
        'city': ['paris', 'berlin', 'amsterdam', 'copenhagen', 'london'],\
        'day': ['yesterday', 'today', 'tomorrow'],\
        'food': ['hamburger', 'salad', 'sandwich', 'fries', 'steak'],\
        'number': ['one', 'two', 'three', 'four', 'five']}
    phrases = [\
        'book me a flight from paris to london tomorrow',\
        'can you meet me in warsaw today',\
        'give me two hamburgers',\
        'can I have a salad please']
    '''
    def test_00_get_glove_vectors(self):
        """ Test getting glove vectors """
        data_path = 'glove'
        vector_file = 'glove.6B.50d.p'
        vector_file_path = os.path.join(data_path, vector_file)
        if os.path.isfile(vector_file_path):
            os.remove(vector_file_path)

        get_glove_vectors()

        assert os.path.isfile(vector_file_path)
    '''
    pickle_file = 'glove/glove.6B.50d.p'
    ner = None

    def __init__(self, *args, **kwargs):
        super(TestNlu, self).__init__(*args, **kwargs)
        if TestNlu.ner is None:
            TestNlu.ner = NamedEntityExtractor(TestNlu.pickle_file)

    def test_02_entity_distances(self):
        """ Test some common entity distances """
        numbers = ['one', 'two', 'three', 'four', 'five']
        cities1 = ['paris', 'london', 'amsterdam', 'madrid', 'shanghai']
        cities2 = ['losangeles', 'sanfrancisco', 'newyork', 'chicago', 'houston']
        singular = ['hamburger', 'hamburger', 'hamburger', 'hamburger', 'hamburger']
        plural = ['hamburgers', 'cheeseburgers', 'cheeseburger', 'salad', 'steak']
        non_matching = ['sunrise', 'tuesday', 'tomorrow', 'bicycle', 'steak']
        can_word = ['can', 'could']
        be_word = ['be', 'is', 'was']
        will_word = ['will', 'would']

        self.print_dist(numbers, numbers)
        self.print_dist(numbers, non_matching)

        self.print_dist(cities1, cities1)
        self.print_dist(cities1, non_matching)

        self.print_dist(cities2, cities2)
        self.print_dist(cities2, non_matching)

        self.print_compare_dist(singular, plural)

        # self.print_dist(can_word, can_word)
        # self.print_dist(be_word, be_word)

    def print_dist(self, entity_list1, entity_list2):
        """ Print the distances between two lists """
        for word1 in entity_list1:
            for word2 in entity_list2:
                self.print_dist_between_words(word1, word2)

    def print_dist_between_words(self, word1, word2):
        """ print dist between words """
        int_word1 = TestNlu.ner.vocab_to_int[word1]
        int_word2 = TestNlu.ner.vocab_to_int[word2]
        euc_dist = NamedEntityExtractor.euclidean_distance(\
            TestNlu.ner.embeddings[int_word1], TestNlu.ner.embeddings[int_word2])
        print("{}:{} euclidean_dist {:.3f}".format(word1, word2, euc_dist))

    def print_compare_dist(self, entity_list1, entity_list2):
        """ Print distances of corresponding items in lists """
        for i, word1 in enumerate(entity_list1):
            self.print_dist_between_words(word1, entity_list2[i])

    '''
    def test_10_named_entity_extraction(self):
        """ Test named entity extraction """
        pickle_file = 'glove/glove.6B.50d.p'
        ner = NamedEntityExtractor(pickle_file)
        for name, entities in TestNlu.entities.items():
            for entity in entities:
                ner.add_named_entity(name, entity)

        found_entities = ner.extract_named_entity(TestNlu.phrases[0].split())
        
        for entity, values in found_entities.items():
            print(entity + " : {}".format(values))
            print()
        
        assert len(found_entities) == 3
    '''

if __name__ == '__main__':
    unittest.main()
