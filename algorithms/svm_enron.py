#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
def svm_al(features_train, features_test, labels_train, labels_test):

    # if use rbf the accuracy will around 61.6%, if use linear the accuracy is around 88%

    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 100]}
    svr = SVC()
    clf = GridSearchCV(svr, parameters)
    # clf = SVC(kernel='rbf', C=1)
    # clf = SVC(kernel='rbf', C=10)
    # clf = SVC(kernel='rbf', C=100)
    # clf = SVC(kernel='rbf', C=1000)
    # clf = SVC(kernel='rbf', C=10000)
    clf.fit(features_train, labels_train)

    clf.predict(features_test)


    return clf
