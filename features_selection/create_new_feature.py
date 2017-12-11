#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../tools/")
sys.path.append("../final_project/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier
from sklearn.naive_bayes import GaussianNB

features_list = ['poi','salary', 'ratio_to_poi', 'ratio_from_poi']

def computeFraction(poi_messages, all_messages):
    if poi_messages == "NaN" or all_messages == "NaN":

        return 0.00
    else:

        poi_messages = float(poi_messages)
        all_messages = float(all_messages)

        fraction = poi_messages / all_messages

        return fraction

def get_data(data_dict):
    new_features_dict = {}
    for name in data_dict:
        data_point = data_dict[name]

        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]

        fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
        # print fraction_from_poi
        data_point["ratio_from_poi"] = fraction_from_poi

        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
        # print fraction_to_poi
        new_features_dict[name] = {"from_poi_to_this_person": round(fraction_from_poi,3),
                             "from_this_person_to_poi": round(fraction_to_poi,3)}
        data_point["ratio_to_poi"] = fraction_to_poi

        new_features_dict[name] = data_point

    return new_features_dict