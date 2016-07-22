#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

clfKNN = KNeighborsClassifier(n_neighbors=9, weights='uniform', algorithm='auto', leaf_size=30, p=1, metric='minkowski', metric_params=None, n_jobs=1)
clfADA = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=0.1, algorithm='SAMME.R', random_state=None)
clfRFC = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
which = raw_input("Enter the classifier to use: ")
#which = "ADA"
if which == "KNN":
    clfKNN.fit(features_train, labels_train)
    print "KNN Accuracy = ", clfKNN.score(features_test, labels_test)
    clf = clfKNN
    prettyPicture(clfKNN, features_test, labels_test)
elif which == "ADA":
    clfADA.fit(features_train, labels_train)
    clf = clfADA
    print "Adaboost Accuracy = ", clfADA.score(features_test, labels_test)
    prettyPicture(clfADA, features_test, labels_test)
else:
    clfRFC.fit(features_train, labels_train)
    clf = clfRFC
    print "RandomForestClassifier Accuracy = ", clfRFC.score(features_test, labels_test)
    prettyPicture(clfRFC, features_test, labels_test)


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass