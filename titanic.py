import sys
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
# from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

# Insert filename
df = pd.read_csv('', header=0)
original_headers = list(df.columns.values)
# df = df._get_numeric_data()
# numeric_headers = list(df.columns.values)
numpy_array = df.as_matrix()
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# print numeric_headers
feature_list = ['Survived', 'Pclass', 'Age', 'Sex', 'SibSp', 'Parch']

# Convert genders to usable values -- 0 for male, and 1 for female
genders = list()
for x in df['Sex']:
    if x == 'male':
        genders.append(0.0)
    elif x == 'female':
        genders.append(1.0)
df['Sex'] = pd.Series(genders)
# print df[['Survived', 'Pclass', 'Age', 'Sex', 'SibSp', 'Parch']]
# Remove 'NaN's
df = df[df['Age'].notnull()]
data = np.array(df[['Survived', 'Pclass', 'Age', 'Sex', 'Parch']], 'float')
# print len(data[0])

#figure out trends in data
# xaxis = 'Survived'
# yaxis = 'Sex'
# plt.scatter(df[xaxis], df[yaxis])
# plt.xlabel(xaxis)
# plt.ylabel(yaxis)
# plt.show()
# sys.exit()

# Create list of labels
predicted_value = feature_list[0]
labels = np.array(data[:, 0 ], 'float')
# Create a list of features
features = list()
for i in range(1, len(data[0])):    
    features.append(np.array(data[:, i], 'float'))
features = np.array(features)

features = np.transpose(features)
# print np.shape(labels)
# print np.shape(features)

# print len(labels), len(features[0])

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3)
# print len(features_train), len(labels_train), len(features_test), len(labels_test)

# clfG = GaussianNB()
# clfG.fit(features_train, labels_train)
# print clfG.score(features_test, labels_test)

# Implementing GridSearchCV with DTC
#param_grid = {'criterion' : ['entropy', 'gini'], 'splitter' : ['best', 'random'], 'min_samples_split' : [2, 3, 4, 5, 6, 7, 8, 9]}
#clfDTC = GridSearchCV(DecisionTreeClassifier(), param_grid)
clfDTC = DecisionTreeClassifier()
clfDTC.fit(features_train, labels_train)
print clfDTC.score(features_test, labels_test)

# Implementing GridSearchCV with SVC
#param_grid = {'C':[1.0, 10.0, 100.0, 500.0], 'kernel':['linear', 'poly', 'sigmoid', 'rbf']}
#clfSVC = GridSearchCV(SVC(), param_grid)
# clfSVC = SVC(C = 500, kernel = 'rbf')
#clfSVC.fit(features_train, labels_train)
#print clfSVC.score(features_test, labels_test)
