# As usual, use a train/test split to get a reliable F1 score from two classifiers, and
# save it the scores in the provided dictionaries.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
X, x_test, y, y_test = cross_validation.train_test_split(X, y)

clf = DecisionTreeClassifier()
clf.fit(X, y)
f1_dt = f1_score(clf.predict(x_test),y_test)
print "Decision Tree F1 score: {:.2f}".format(f1_dt)

clf = GaussianNB()
clf.fit(X, y)
f1_nb = f1_score(clf.predict(x_test),y_test)
print "GaussianNB F1 score: {:.2f}".format(f1_nb)

F1_scores = {
 "Naive Bayes": f1_nb,
 "Decision Tree": f1_dt
}