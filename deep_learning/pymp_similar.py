import numpy as np
from six.moves import cPickle as pickle
from time import time

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

def data_diff(data1, data2):
  return np.sum(np.abs(np.subtract(data1, data2)))

def is_similar(data1, data2, label1, label2, similar_threshold = 0):
  diff = data_diff(data1, data2)   
  similar = diff <= similar_threshold
  if (similar):
    print (label1, " is similar to ", label2, " with diff ", diff, " and thresold ", similar_threshold)
  return similar

import pymp

def overlap_count(data1, data2, labels1, labels2, threshold = 0):
  n_data1 = len(data1)
  n_data2 = len(data2)
  n_total = n_data1 * n_data2
  ex_counters = pymp.shared.array((2,), dtype='uint32')
  ex_data1 = pymp.shared.array(data1.shape, dtype='float32')
  ex_data2 = pymp.shared.array(data2.shape, dtype='float32')
  ex_labels1 = pymp.shared.array(labels1.shape, dtype='int32')
  ex_labels2 = pymp.shared.array(labels2.shape, dtype='int32')
  
  for i in xrange(n_data1):
    if (ex_labels1[i] == 8):
      continue
    with pymp.Parallel(30) as p: 
      for j in p.range(n_data2):
        if (ex_labels1[i] == ex_labels2[j] and is_similar(ex_data1[i], ex_data2[j], ex_labels1[i], ex_labels2[j])):
          with p.lock:
            ex_counters[0] += 1
            print 'found ', ex_counters[0], ' overlaps at ', i, ',', j
      with p.lock:
        ex_counters[1] += 1
        if (ex_counters[1] % 1000 == 0):
          print 'Made ', ex_counters[1], ' processes out of ', n_data1
  return ex_counters[0]

t0 = time()
print ("Train and Validate Overlap: ", overlap_count(train_dataset, valid_dataset, train_labels, valid_labels))
print ("Time elapsed ", round(time() - t0, 3))
t0 = time()
print ("Train and Test Overlap:     ", overlap_count(train_dataset, test_dataset, train_labels, test_labels))
print ("Time elapsed ", round(time() - t0, 3))
t0 = time()
print ("Validate and Test Overlap:  ", overlap_count(valid_dataset, test_dataset, valid_labels, test_labels))        
print ("Time elapsed ", round(time() - t0, 3))
