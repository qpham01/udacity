"""
Download and read text8 dataset
"""
from urllib.request import urlretrieve
import zipfile
import pickle
from os.path import isfile, isdir, getsize
from utils import DLProgress
import numpy as np
from tqdm import tqdm

DATASET_FOLDER_PATH = 'glove'
DATASET_FILENAME = 'glove.6B.zip'
DATASET_NAME = 'Glove 6 Billion Tokens Dataset'

def get_glove_vectors(vector_count=50):
    """
    Get Stanford NLP's Glove vectors

    vector_count should be in [50, 100, 200, 300]
    """
    if not isfile(DATASET_FILENAME):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc=DATASET_NAME) as pbar:
            urlretrieve(
                'http://nlp.stanford.edu/data/glove.6B.zip',
                DATASET_FILENAME,
                pbar.hook)

    if not isdir(DATASET_FOLDER_PATH):
        with zipfile.ZipFile(DATASET_FILENAME) as zip_ref:
            zip_ref.extractall(DATASET_FOLDER_PATH)

    file_name = '{}/glove.6B.{}d.txt'.format(DATASET_FOLDER_PATH, vector_count)
    print("Loading file {}, size {} bytes".format(file_name, getsize(file_name)))
    lines = []
    with open(file_name, encoding='utf-8') as fread:
        line_number = 0
        line = fread.readline()
        while line:
            line_number += 1
            lines.append(line)
            try:
                line = fread.readline()
            except UnicodeDecodeError as err:
                print("Skipped line {} due to Unicode error".format(line_number))
                continue



    print("Building vectors")
    vocab_size = len(lines)
    all_vectors = np.zeros((vocab_size, vector_count), dtype=float)
    vocab_to_int = {}
    int_to_vocab = {}
    vocab = []
    error_count = 0
    for index, line in enumerate(tqdm(lines)):
        line_count = index + 1
        tokens = line.split()
        if len(tokens) != vector_count + 1:
            print("Only {} tokens on line {}: {}".format(len(tokens), line_count, line))
            continue
        for i in range(vector_count):
            try:
                all_vectors[index, i] = float(tokens[i + 1])
            except IndexError as err:
                print("line {}:{} Index {} is out of range".format(line_count, tokens[0], i))
                error_count += 1
            except ValueError as err:
                print("Skipped line {} due to value error {}".format(line_number, err))
                continue
        word = tokens[0]
        vocab_to_int[word] = index
        int_to_vocab[index] = word
        vocab.append(word)

    print("Error count is {}".format(error_count))

    embed_data = {}
    embed_data["embeddings"] = all_vectors
    embed_data["vocab"] = vocab
    embed_data["vocab_to_int"] = vocab_to_int
    embed_data["int_to_vocab"] = int_to_vocab

    pickle_file_name = '{}/glove.6B.{}d.p'.format(DATASET_FOLDER_PATH, vector_count)
    with open(pickle_file_name, 'wb') as fwrite:
        pickle.dump(embed_data, fwrite)
