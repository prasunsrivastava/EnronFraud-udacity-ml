#!/usr/bin/python

import sys
import pickle
from copy import deepcopy

from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.cross_validation import train_test_split

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
features_list += binary_features

# create a feature that shows how much interaction did the person have
# with poi.
create_email_features(my_data, features_list)

features_list = [target_label] + features_list

### Store to my_dataset for easy export below.
my_dataset = my_data

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# prepare training and test set
X_train, X_test, y_train, y_test = train_test_split(features, labels, 
	test_size=0.2, stratify=labels, random_state=42)

# train a decision tree classifier
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(random_state=2016)

criteria = ['gini', 'entropy']
max_depth = [1, 3, 5, 7]
max_features = [5, 6, 10, 20, 39]
min_samples_leaf = [1, 3, 5]
class_weight = ['balanced']

param_grid = {'criterion': criteria,
              'max_depth': max_depth,
              'max_features': max_features,
              'min_samples_leaf': min_samples_leaf,
              'class_weight': class_weight}

estimator_dt = GridSearchCV(dt_clf, param_grid, cv=5)

estimator_dt.fit(X_train, y_train)

dt_predictions = estimator_dt.predict(X_test)

print "Decision Tree Classification Report:\n"
print classification_report(y_test, dt_predictions)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf_model= RandomForestClassifier(class_weight='balanced', random_state=999)

num_estimators = [5, 10, 20, 50]
leaf_size = [1, 3, 5]
max_features = [5, 6, 10, 20]
param_grid = {'n_estimators': num_estimators,
              'max_features': max_features,
              'min_samples_leaf':leaf_size}

rf_estimator = GridSearchCV(rf_model, param_grid, scoring="f1_macro", cv=10)

rf_estimator.fit(X_train, y_train)

print "Random Forest Classification Report:\n"
print classification_report(y_test, rf_estimator.predict(X_test))

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(random_state=42, class_weight='balanced')

C = [10**num for num in range(-6, 6)]
param_grid = {'C':C, 'penalty':['l1', 'l2']}

estimator_lr = GridSearchCV(clf_lr, param_grid, cv=10, scoring='f1_macro')

estimator_lr.fit(X_train, y_train)

print "Logistic Regression Classification Report:\n"
print classification_report(y_test, estimator_lr.predict(X_test))
print
print estimator_lr.best_params_

print "\nFeatures Not used in model:\n"
for feature, coef in zip(features_list[1:], estimator_lr.best_estimator_.coef_[0]):
    if coef == 0.0:
        print feature
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# classifiers already tuned in the earlier task. So, only training with the full data and best params. The 
# whole process of tuning and selecting written up in `Model Selection.ipynb` notebook.
# train logistic regression classifier here with full training set and the best parameters obtained earlier
# when cross-validated and tuned.
clf = estimator_lr.best_estimator_
clf.fit(X_train, y_train)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)