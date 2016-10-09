#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
import numpy    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC

klinear = 'linear'
krbf = 'rbf'

#Cs = [10, 100, 1000, 10000]
Cs = [ 10000 ]

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

for Cv in Cs:
    print "==============================="
    print "C: ", Cv
    clf = SVC(kernel=krbf, C=Cv)

    t0 = time()
    clf.fit(features_train, labels_train)
    print "train time: ", round(time() - t0, 3), " seconds"

    t0 = time()
    pred = clf.predict(features_test)
    print "predict time: ", round(time() - t0, 3), " seconds"

    accuracy = clf.score(features_test, labels_test)
    print "accuracy: ", accuracy, " for C: ", Cv

    unique, counts = numpy.unique(pred, return_counts=True)
    counts = dict(zip(unique, counts))
    print "Sara email count:  ", counts[0]
    print "Chris email count: ", counts[1]
    #elems = [10, 26, 50]
    #for i in elems:
    #    print i, ": ", pred[i]
#########################################################