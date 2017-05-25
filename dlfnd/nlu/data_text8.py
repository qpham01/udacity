"""
Download and read text8 dataset
"""
from urllib.request import urlretrieve
import zipfile
from os.path import isfile, isdir
from utils import DLProgress

DATASET_FOLDER_PATH = 'data'
DATASET_FILENAME = 'text8.zip'
DATASET_NAME = 'Text8 Dataset'

def get_text8():
    """ Get Matt Mahoney's text8 cleaned up wikipedia article text """
    if not isfile(DATASET_FILENAME):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc=DATASET_NAME) as pbar:
            urlretrieve(
                'http://mattmahoney.net/dc/text8.zip',
                DATASET_FILENAME,
                pbar.hook)

    if not isdir(DATASET_FOLDER_PATH):
        with zipfile.ZipFile(DATASET_FILENAME) as zip_ref:
            zip_ref.extractall(DATASET_FOLDER_PATH)

    with open('data/text8') as fread:
        text = fread.read()

    return text
