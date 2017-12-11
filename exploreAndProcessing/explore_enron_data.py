#!/usr/bin/python
# -*- coding: utf-8 -*-

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

poi_num = 0

# 数据点的长度
print len(enron_data)

# POI的数量
for k, v in enron_data.items():
    if v['poi'] == True:
        poi_num += 1

print "POI的数量:" + str(poi_num)

# 查找每个特征缺失值得数量
outlier_dic = {}
num = 0

for k, v in enron_data.items():
    for kk, vv in v.items():
        if vv == 'NaN':
            if outlier_dic.has_key(kk):
                outlier_dic[kk] += 1
            else:
                outlier_dic[kk] = 1

for k,v in outlier_dic.items():

   print k + ": " + str(v)
