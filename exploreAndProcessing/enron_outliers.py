#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import sys
import matplotlib.pyplot

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


#在分析出异常值以后，清理异常值
def clean_outlier(data_dict):
    data_dict.pop('TOTAL', 0)
    return data_dict



# data = featureFormat(data_dict, features)
# for point in data:
#         salary = point[0]
#         bonus = point[1]
#         matplotlib.pyplot.scatter(salary, bonus)
#
# matplotlib.pyplot.xlabel("salary")
# matplotlib.pyplot.ylabel("bonus")
# matplotlib.pyplot.title('Salary vs Bonus')
# matplotlib.pyplot.show()
