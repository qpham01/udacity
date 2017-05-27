"""
Code for working with named entities
"""
import pickle
import numpy as np

class NamedEntityExtractor:
    """ Encapsulates training of named entities """
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as fread:
            embed_data = pickle.load(fread)
        self.embeddings = embed_data["embeddings"]
        self.vocab = embed_data["vocab"]
        self.vocab_to_int = embed_data["vocab_to_int"]
        self.int_to_vocab = embed_data["int_to_vocab"]
        self.named_entities = dict()
        if "named_entities" in embed_data:
            self.named_entities = embed_data["named_entities"]
        self.named_entity_embeddings = dict()
        if "named_entity_embeddings" in embed_data:
            self.named_entities = embed_data["named_entity_embeddings"]

    def add_named_entity(self, name, entity):
        """
        Add a named entity to the training set

        Parameters:
        -----------
        name: a string that names the entity
        entity:  list of word tokens comprising the entity

        Returns:
        --------
        None
        """
        if not name in self.named_entities:
            self.named_entities[name] = []
        self.named_entities[name].append(entity)

        if not name in self.named_entity_embeddings:
            self.named_entity_embeddings[name] = []
        embedding = self.embeddings[self.vocab_to_int[entity]]
        self.named_entity_embeddings[name].append(embedding)

    def extract_named_entity(self, tokens, threshold=0.2):
        """ Extract a named entity by cosine distance """
        # for each token, find its word vector
        token_embeddings = []
        for token in tokens:
            if token in self.vocab:
                int_token = self.vocab_to_int[token]
                embedding = self.embeddings[int_token]
                token_embeddings.append((token, embedding))

        candidates = {}
        for name in self.named_entities:
            entity_list = self.named_entity_embeddings[name]
            found_entities = self.find_matching_entities(token_embeddings, \
                self.named_entities[name], entity_list, threshold)
            if len(found_entities) > 0:
                if not name in candidates:
                    candidates[name] = []
                candidates[name].extend(found_entities)
        return candidates

    def find_matching_entities(self, tokens, entities, entity_embeddings, threshold, \
        method='cosine distance'):
        """
        Find the best matching entity from a named entity list given word tokens from phrase.

        Parameters:
        -----------
        tokens:  A list of 2-tuples (token, embeddings) to search for entities in
        entity_list:  A list of pre-defined entity embeddings
        threshold:  The metric for determine entity matching.  Its use depends on the
            method parameter.
        method:  The string identifying the matching method to use.  Only 'cosine distance'
            is currently supported.

        Returns:
        --------
        A list of matching (score, entity tokens) 2-tuples, which could be empty if no match
        was found.
        """
        found = []
        for token, vectors in tokens:
            for i, entity in enumerate(entity_embeddings):
                if method == 'cosine distance':
                    cosines = np.dot(entity, vectors)
                    score = np.mean(cosines)
                    print("Score between {} and {} is {}".format(token, entities[i], score))
                else:
                    raise "Unknown entity matching method {}".format(method)
                if score < threshold:
                    found.append((score, token))
        return found
