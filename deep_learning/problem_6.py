# Problem #6 Fitting with off-the-self linear regression model

import numpy as np
from time import time
from read_data import *
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

n_samples = [50, 100, 1000, 5000, len(train_dataset)]

def random_sample(data, labels, sample_size):
  indices = np.random.choice(range(len(data)), size=sample_size)
  return np.take(data, indices,axis=0), np.take(labels, indices)

def linear_reshape(data):
  #delete index column
  data = np.delete(data, 0, 1)
  #reshape to 1-D rows of pixels
  return np.reshape(data, (len(data), 27*28))

for n in n_samples:
  train_rdata, train_rlabels = random_sample(train_dataset, train_labels, n)
  train_rdata = linear_reshape(train_rdata)  
  t0 = time() 
  reg.fit(train_rdata, train_rlabels)
  print "Fitted ", n, "samples in: ", round(time() - t0, 3), "seconds"

  test_reg_data = linear_reshape(test_dataset)
  print "Score for ", n, "samples: ", reg.score(test_reg_data, test_labels)