#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

sys.path.append("../tools/")
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
def dt_al(features_train, features_test, labels_train, labels_test):

    clf = tree.DecisionTreeClassifier(min_samples_split=2)
    # clf = tree.DecisionTreeClassifier(min_samples_split=4)
    # clf = tree.DecisionTreeClassifier(min_samples_split=8)
    # clf = tree.DecisionTreeClassifier(min_samples_split=16)
    # clf = tree.DecisionTreeClassifier(min_samples_split=32)
    # clf = tree.DecisionTreeClassifier(min_samples_split=64)
    clf = clf.fit(features_train, labels_train)

    clf.predict(features_test)
    #
    # acc = accuracy_score(pred, labels_test)

    return clf
