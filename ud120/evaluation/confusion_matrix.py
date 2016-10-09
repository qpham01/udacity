# In this exercise, we'll use the Titanic dataset as before, train two classifiers and
# look at their confusion matrices. Your job is to create a train/test split in the data
# and report the results in the dictionary at the bottom.

import numpy as np
import pandas as pd

# Load the dataset
from sklearn import datasets

X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
Y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
X_train, x, Y_train, y = cross_validation.train_test_split(X, Y)

clf = DecisionTreeClassifier()
clf.fit(X_train,Y_train)
dtree_confusion = confusion_matrix(clf.predict(x),y)
print "Confusion matrix for this Decision Tree:\n", dtree_confusion

clf = GaussianNB()
clf.fit(X_train,Y_train)
gaussian_confusion = confusion_matrix(clf.predict(x),y);
print "GaussianNB confusion matrix:\n",gaussian_confusion

#TODO: store the confusion matrices on the test sets below

confusions = {
 "Naive Bayes": gaussian_confusion,
 "Decision Tree": dtree_confusion
}