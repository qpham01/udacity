"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)

view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

from collections import Counter

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    punctuations = ['.', ',', '"', ';', '!', '?', '(', ')', '--', '\n']
    tokens = ['<EOS>', '<COMMA>', '<DQUOTE>', '<SEMICOLON>', '<BANG>', '<QUESTION>', '<PAREN1>', '<PAREN2>', '<DDASH>', '<NEWLINE>']
    return {x: y for (x,y) in zip(punctuations, tokens)}

def create_lookup_tables(text):
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {idx: word for idx, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: idx for idx, word in enumerate(sorted_vocab)}
    vocab_size = len(vocab_to_int)    

    return vocab_to_int, int_to_vocab

token_source_text = source_text
token_target_text = target_text

# Convert punctuations
token_dict = token_lookup()
for key, token in token_dict.items():
    token_source_text = token_source_text.replace(key, '{}'.format(token))
    token_target_text = token_target_text.replace(key, '{}'.format(token))


# Split text along white space before creating integer look up tables
#token_source_text = token_source_text.lower()
token_source_text = token_source_text.split()

#token_target_text = token_target_text.lower()
token_target_text = token_target_text.split()

source_vocab_to_int, source_int_to_vocab = create_lookup_tables(token_source_text)

target_vocab_to_int, target_int_to_vocab = create_lookup_tables(token_target_text)

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function
    token_source_sentences = source_text.split('\n')
    source_id_text = []
    for sentence in token_source_sentences:
        # print("source: ", sentence)
        words = sentence.split()
        id_text = [source_vocab_to_int[word] for word in words]
        source_id_text.append(id_text)

    token_target_text = target_text.replace('.', '. <EOS>')
    token_target_sentences = token_target_text.split('\n')
    target_id_text = []
    for sentence in token_target_sentences:
        # print("target: ", sentence)
        words = sentence.split()
        id_text = [target_vocab_to_int[word] for word in words]
        target_id_text.append(id_text)
    
    return source_id_text, target_id_text

import collections
import itertools

def test_text_to_ids(text_to_ids):
    test_source_text = 'new jersey is sometimes quiet during autumn , and it is snowy in april .\nthe united states is usually chilly during july , and it is usually freezing in november .\ncalifornia is usually quiet during march , and it is usually hot in june .\nthe united states is sometimes mild during june , and it is cold in september .'
    test_target_text = 'new jersey est parfois calme pendant l\' automne , et il est neigeux en avril .\nles états-unis est généralement froid en juillet , et il gèle habituellement en novembre .\ncalifornia est généralement calme en mars , et il est généralement chaud en juin .\nles états-unis est parfois légère en juin , et il fait froid en septembre .'

    test_source_text = test_source_text.lower()
    test_target_text = test_target_text.lower()

    source_vocab_to_int, source_int_to_vocab = helper.create_lookup_tables(test_source_text)
    target_vocab_to_int, target_int_to_vocab = helper.create_lookup_tables(test_target_text)

    test_source_id_seq, test_target_id_seq = text_to_ids(test_source_text, test_target_text, source_vocab_to_int, target_vocab_to_int)

    len_test_source_id_seq = len(test_source_id_seq)
    len_test_source_text_split = len(test_source_text.split('\n'))
    assert len(test_source_id_seq) == len(test_source_text.split('\n')),\
        'source_id_text has wrong length {}, it should be {}.'.format(len_test_source_id_seq, len(test_source_text.split('\n')))
    assert len(test_target_id_seq) == len(test_target_text.split('\n')), \
        'target_id_text has wrong length, it should be {}.'.format(len(test_target_text.split('\n')))

    target_not_iter = [type(x) for x in test_source_id_seq if not isinstance(x, collections.Iterable)]
    assert not target_not_iter,\
        'Element in source_id_text is not iteratable.  Found type {}'.format(target_not_iter[0])
    target_not_iter = [type(x) for x in test_target_id_seq if not isinstance(x, collections.Iterable)]
    assert not target_not_iter, \
        'Element in target_id_text is not iteratable.  Found type {}'.format(target_not_iter[0])

    source_changed_length = [(words, word_ids)
                             for words, word_ids in zip(test_source_text.split('\n'), test_source_id_seq)
                             if len(words.split()) != len(word_ids)]
    assert not source_changed_length,\
        'Source text changed in size from {} word(s) to {} id(s): {}'.format(
            len(source_changed_length[0][0].split()), len(source_changed_length[0][1]), source_changed_length[0][1])

    target_missing_end = [word_ids for word_ids in test_target_id_seq if word_ids[-1] != target_vocab_to_int['<EOS>']]
    assert not target_missing_end,\
        'Missing <EOS> id at the end of {}'.format(target_missing_end[0])

    target_bad_size = [(words.split(), word_ids)
                       for words, word_ids in zip(test_target_text.split('\n'), test_target_id_seq)
                       if len(word_ids) != len(words.split()) + 1]
    assert not target_bad_size,\
        'Target text incorrect size.  {} should be length {}'.format(
            target_bad_size[0][1], len(target_bad_size[0][0]) + 1)

    source_bad_id = [(word, word_id)
                     for word, word_id in zip(
                        [word for sentence in test_source_text.split('\n') for word in sentence.split()],
                        itertools.chain.from_iterable(test_source_id_seq))
                     if source_vocab_to_int[word] != word_id]
    assert not source_bad_id,\
        'Source word incorrectly converted from {} to id {}.'.format(source_bad_id[0][0], source_bad_id[0][1])

    target_bad_id = [(word, word_id)
                     for word, word_id in zip(
                        [word for sentence in test_target_text.split('\n') for word in sentence.split()],
                        [word_id for word_ids in test_target_id_seq for word_id in word_ids[:-1]])
                     if target_vocab_to_int[word] != word_id]
    assert not target_bad_id,\
        'Target word incorrectly converted from {} to id {}.'.format(target_bad_id[0][0], target_bad_id[0][1])

    _print_success_message()


def _print_success_message():
    print('Tests Passed')

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
test_text_to_ids(text_to_ids)

import helper
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
helper.preprocess_and_save_data(source_path, target_path, text_to_ids)