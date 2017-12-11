#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")
sys.path.append("../exploreAndProcessing/")
sys.path.append("../features_selection/")
sys.path.append("../algorithms/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from enron_outliers import clean_outlier
from create_new_feature import get_data
from find_signature import split_train_test
from nb_enron import bayes_al
from dt_enron import dt_al
from svm_enron import svm_al
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'exercised_stock_options',
                 'bonus',
                 'shared_receipt_with_poi',
                 'total_stock_value']

# You will need to use more features
# features_list = ['poi', 'salary']  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

###Remove outliers
data_dict = clean_outlier(data_dict)

# ### Task 3: Create new feature(s)
# ### Store to my_dataset for easy export below.
my_dataset = get_data(data_dict)

# my_dataset = data_dict

## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = split_train_test(features, labels)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# clf = bayes_al(features_train, features_test, labels_train, labels_test)
# clf = svm_al(features_train, features_test, labels_train, labels_test)
clf = dt_al(features_train, features_test, labels_train, labels_test)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
