import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



########################## DECISION TREE #################################

from sklearn import tree
from sklearm import metrics

cls2 = tree.DecisionTreeClassifier(min_sample_split=2)
cls2.fit(features_train, labels_train)

pred = cls2.predict(features_test, labels_test)
acc_min_samples_split_2 = metrics.accuracy_score(pred, labels_test)

cls50 = tree.DecisionTreeClassifier(min_sample_split=50)
cls50.fit(features_train, labels_train)

pred = cls50.predict(features_test, labels_test)
acc_min_samples_split_50 = metrics.accuracy_score(pred, labels_test)


def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}