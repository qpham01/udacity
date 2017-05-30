""" Test natural language understanding """
import unittest
from named_entity import NamedEntityExtractor

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

    def test_01_named_entity_extraction(self):
        """ Test named entity extraction """
        pickle_file = 'data/text8_embeddings.p'
        ner = NamedEntityExtractor(pickle_file)
        for name, entities in TestNlu.entities.items():
            for entity in entities:
                ner.add_named_entity(name, entity)

        found_entities = ner.extract_named_entity(TestNlu.phrases[0].split())
        for entity, values in found_entities.items():
            print(entity + " : {}".format(values))
            print()
        assert len(found_entities) == 3

if __name__ == '__main__':
    unittest.main()
