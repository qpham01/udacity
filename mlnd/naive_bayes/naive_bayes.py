import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
# X is array of features and Y is array of labels
clf.fit(X, Y)  # train and fit are the same thing

print (clf.predict([[-0.8, -1]]))
clf