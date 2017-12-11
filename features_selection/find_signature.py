#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import MinMaxScaler

numpy.random.seed(42)

### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier

features_list = []
labels_list = []


def split_train_test(features, labels):

    # 特征缩放
    scaler = MinMaxScaler()

    rescale_weights_features = scaler.fit_transform(features)
    rescale_weights_labels = scaler.fit_transform(labels)

    features_train, features_test, labels_train, labels_test = train_test_split(rescale_weights_features, rescale_weights_labels,
                                                                                test_size=0.3,
                                                                                 random_state=42)
    #特征选择
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train, labels_train)

    # print selector.scores_

    features_train_transformed = selector.transform(features_train)
    features_test_transformed = selector.transform(features_test)


    return features_train_transformed, features_test_transformed, labels_train, labels_test


    # 在没有分离训练和测试数据时的预测准确率
    # clf = tree.DecisionTreeClassifier()
    # clf.fit(features_train, labels_train)
    # pred = clf.predict(features)
    # print accuracy_score(pred, labels)


