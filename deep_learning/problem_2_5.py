"""
Problem #5
By construction, this dataset might contain a lot of overlapping samples, including training data that's also 
contained in the validation and test set! Overlap between training and test can skew the results if you expect 
to use your model in an environment where there is never an overlap, but are actually ok if you expect to see 
training samples recur when you use it. Measure how much overlap there is between training, validation and test
samples.

Optional questions:

    What about near duplicates between datasets? (images that are almost identical)
    Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
"""

import numpy as np
from time import time
from read_data import *

def data_diff(data1, data2):
  return np.sum(np.abs(np.subtract(data1, data2)))

def is_similar(data1, data2, label1, label2, similar_threshold = 0):
  diff = data_diff(data1, data2)   
  similar = diff <= similar_threshold
  if (similar):
    print (label1, " is similar to ", label2, " with diff ", diff, " and thresold ", similar_threshold)
  return similar

def overlap_count(data1, data2, labels1, labels2, threshold = 0):
  n_data1 = len(data1)
  n_data2 = len(data2)
  n_total = n_data1 * n_data2
  n_overlap = 0
  n_compare = 0
  for i in xrange(n_data1):
    if (labels1[i] == 8):
      continue
    for j in xrange(n_data2):
      if (labels1[i] == labels2[j] and is_similar(data1[i], data2[j], labels1[i], labels2[j])):
        n_overlap += 1
        print ("found ", n_overlap, " overlaps at ", i, ",", j)
    n_compare += 1
    if (n_compare % 1000 == 0):
        print("Made ", n_compare, " processes out of ", n_data1)
  return n_overlap

t0 = time()
print ("Train and Validate Overlap: ", overlap_count(train_dataset, valid_dataset, train_labels, valid_labels))
print ("Time elapsed ", round(time() - t0, 3))
t0 = time()
print ("Train and Test Overlap:     ", overlap_count(train_dataset, test_dataset, train_labels, test_labels))
print ("Time elapsed ", round(time() - t0, 3))
t0 = time()
print ("Validate and Test Overlap:  ", overlap_count(valid_dataset, test_dataset, valid_labels, test_labels))        
print ("Time elapsed ", round(time() - t0, 3))