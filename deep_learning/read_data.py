import numpy as np
from six.moves import cPickle as pickle

pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'rb')
  data = pickle.load(f)
  train_dataset = data['train_dataset']
  valid_dataset = data['valid_dataset']
  test_dataset = data['test_dataset']
  train_labels = data['train_labels']
  valid_labels = data['valid_labels']
  test_labels = data['test_labels']
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise