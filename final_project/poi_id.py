#!/usr/bin/python

import sys
import pickle
from copy import deepcopy
import numpy as np

from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from util import create_binary_missing_features, create_email_features

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
target_label = 'poi'

email_features = ['to_messages',
                  'from_messages',
                  'from_this_person_to_poi',
                  'from_poi_to_this_person',
                  'shared_receipt_with_poi']

finance_features = ['salary',
                    'deferral_payments',
                    'total_payments',
                    'exercised_stock_options',
                    'bonus',
                    'restricted_stock',
                    'restricted_stock_deferred',
                    'total_stock_value',
                    'expenses',
                    'loan_advances',
                    'other',
                    'director_fees',
                    'deferred_income',
                    'long_term_incentive']

features_list = email_features + finance_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']

for outlier in outliers:
    data_dict.pop(outlier, 0)

### Task 3: Create new feature(s)

my_data = deepcopy(data_dict)

# create binary features that keep track if a value was missing or not
binary_features = create_binary_missing_features(my_data, features_list)
binary_features = list(binary_features)

# create a feature that shows how much interaction did the person have
# with poi.
create_email_features(my_data, features_list)

# initialize the selected feature list obtained during model tuning
selected_features = ['to_messages',
                     'from_messages',
                     'from_this_person_to_poi',
                     'from_poi_to_this_person',
                     'shared_receipt_with_poi',
                     'salary',
                     'deferral_payments', 
                     'total_payments', 
                     'exercised_stock_options',
                     'bonus', 
                     'restricted_stock', 
                     'total_stock_value', 
                     'expenses',
                     'other', 
                     'deferred_income', 
                     'long_term_incentive', 
                     'missing_bonus',
                     'missing_exercised_stock_options', 
                     'missing_to_messages',
                     'interaction_with_poi']

features_list = [target_label] + selected_features

### Store to my_dataset for easy export below.
my_dataset = my_data

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4 & Task 5
### Model selection and parameter tuning are done separately
### in the notebook `Model Selection.ipynb`. Hence, here 
### training the model on best parameters obtained during
### parameter tuning. However, to illustrate the cross-validation
### performance and the performance on the final test data,
### 10-fold Stratified Cross Validation will be used.
### Please name your classifier clf for easy export below.

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

labels, features = np.array(labels), np.array(features)

# prepare train and test data
features_train, features_test, labels_train, labels_test = train_test_split(features, 
                                                                            labels,
                                                                            test_size=0.2,
                                                                            stratify=labels, 
                                                                            random_state=42)

clf = make_pipeline(StandardScaler(),
	                LogisticRegression(penalty='l1', C=10, class_weight='balanced', random_state=42))

folds = StratifiedKFold(labels_train, n_folds=10, shuffle=True, random_state=42)

precision = []
recall = []

for train_idx, test_idx in folds:
    
    train_X = features_train[train_idx]
    train_y = labels_train[train_idx]
    test_X = features_train[test_idx]
    test_y = labels_train[test_idx]

    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    
    precision.append(precision_score(test_y, pred))
    recall.append(recall_score(test_y, pred))\

print "Cross-Validated Average Precision:", round(np.mean(precision), 2)
print "Cross-Validated Average Recall:", round(np.mean(recall), 2), "\n"

# fit the model on whole of training data
clf.fit(features_train, labels_train)

# print classification report for the test data
print classification_report(labels_test, clf.predict(features_test))


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)