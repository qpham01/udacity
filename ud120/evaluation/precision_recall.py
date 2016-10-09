# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
Y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
X_train, x, Y_train, y = cross_validation.train_test_split(X, Y)

clf1 = DecisionTreeClassifier()
clf1.fit(X_train, Y_train)
recall1 = recall(y,clf1.predict(x))
precision1 = precision(y,clf1.predict(x))
print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(recall1, precision1)

clf2 = GaussianNB()
clf2.fit(X_train, Y_train)
recall2 = recall(y,clf2.predict(x))
precision2 = precision(y,clf2.predict(x))
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(recall2, precision2)

results = {
  "Naive Bayes Recall": recall2,
  "Naive Bayes Precision": precision2,
  "Decision Tree Recall": recall1,
  "Decision Tree Precision": precision1
}