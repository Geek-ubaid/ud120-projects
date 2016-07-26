#!/usr/bin/python

import math
import pickle
import sys

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, main
from sklearn.metrics.scorer import f1_scorer

sys.path.append("../tools/")


# ## Task 1: Select what features you'll use.
# ## features_list is a list of strings, each of which is a feature name.
# ## The first feature must be "poi".
features_list = ['poi', 'total_payments', 'from_this_person_to_poi', 'from_poi_to_this_person', 'exercised_stock_options', 'long_term_incentive', 'expenses']  # You will need to use more features

# ## Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# ## Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
# ## Task 3: Create new feature(s)
for key in data_dict.keys():
    from_ratio = float(data_dict[key]["from_this_person_to_poi"]) / float(data_dict[key]["from_messages"])
    to_ratio = float(data_dict[key]["from_poi_to_this_person"]) / float(data_dict[key]["to_messages"])
    data_dict[key]["from_ratio"] = "NaN" if math.isnan(from_ratio) else from_ratio
    data_dict[key]["to_ratio"] = "NaN" if math.isnan(to_ratio) else to_ratio
# ## Store to my_dataset for easy export below.
my_dataset = data_dict

# ## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(features)

param_grid = {'n_components' : [1, 2, 3, 4, 5, 6]}

pca = GridSearchCV(PCA(), param_grid)
pca = pca.fit(features)
features = pca.transform(features)

# ## Task 4: Try a varity of classifiers
# ## Please name your classifier clf for easy export below.
# ## Note that if you want to do PCA or other multi-stage operations,
# ## you'll need to use Pipelines. For more info:
# ## http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# param_grid = {'criterion' : ['entropy', 'gini'], 'splitter' : ['best', 'random'], 'min_samples_split' : [2, 3, 4, 5, 6, 7, 8, 9]}
# clf = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = f1_scorer)
clf = DecisionTreeClassifier(criterion="gini")
clf = clf.fit(features, labels)
# clf = SVC()
# clf.fit(features, labels)


# ## Task 5: Tune your classifier to achieve better than .3 precision and recall 
# ## using our testing script. Check the tester.py script in the final project
# ## folder for details on the evaluation method, especially the test_classifier
# ## function. Because of the small size of the dataset, the script uses
# ## stratified shuffle split cross validation. For more info: 
# ## http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# ## Task 6: Dump your classifier, dataset, and features_list so anyone can
# ## check your results. You do not need to change anything below, but make sure
# ## that the version of poi_id.py that you submit can be run on its own and
# ## generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

main()
