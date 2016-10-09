#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################

from time import time
from sklearn.metrics import accuracy_score

def k_nearest_neighbors():
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier() 
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc_knn = accuracy_score(labels_test, pred)
    try:
        prettyPicture(clf, features_test, labels_test, "test_knn.png")
    except NameError:
        pass
    return acc_knn

t0 = time()
print "k nearest neighbor accuracy: ", k_nearest_neighbors()
print "knn time: ", round(time() - t0, 3), " seconds"

def adaboost():
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier() 
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc_ada = accuracy_score(labels_test, pred)
    try:
        prettyPicture(clf, features_test, labels_test, "test_ada.png")
    except NameError:
        pass
    return acc_ada

t0 = time()
print "adaboost accuracy: ", adaboost()
print "adaboost time: ", round(time() - t0, 3), " seconds"

def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(criterion="gini", min_samples_split=10) 
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc_rf = accuracy_score(labels_test, pred)
    try:
        prettyPicture(clf, features_test, labels_test, "test_rf.png")
    except NameError:
        pass
    return acc_rf

t0 = time()
print "random forest accuracy: ", random_forest()
print "random forest time: ", round(time() - t0, 3), " seconds"

def svm():
    from sklearn.svm import SVC
    clf = SVC(C=100000.0, kernel="rbf")
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc_svm = accuracy_score(labels_test, pred)
    try:
        prettyPicture(clf, features_test, labels_test, "test_svm.png")
    except NameError:
        pass
    return acc_svm

t0 = time()
print "SVM accuracy: ", svm()
print "SVM time: ", round(time() - t0, 3), " seconds"

def naive_bayes():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc_nb = accuracy_score(labels_test, pred)
    try:
        prettyPicture(clf, features_test, labels_test, "test_nb.png")
    except NameError:
        pass
    return acc_nb

t0 = time()
print "naive bayes accuracy: ", naive_bayes()
print "naive bayes time: ", round(time() - t0, 3), " seconds"