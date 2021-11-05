import os

import numpy as np

train_file = open(os.path.join('..', 'data', 'level_2', 'train.csv'), 'r')

lines = train_file.readlines()

num_samples = int(lines[0].strip())

X = []
y = []
results = []

for n in range(num_samples):
    sample_line = lines[1 + n].strip()
    label_line = lines[1 + num_samples + n].strip()
    sample = sample_line.split(',')
    sample = [int(x) for x in sample]
    label = int(label_line)
    X.append(sample)
    y.append(label)

print("data ready")

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

#X, y = shuffle(X, y, random_state=42)
print("scaling...")
X = StandardScaler().fit_transform(X)

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Create a classifier: a support vector classifier
#clf = svm.SVC(gamma=0.001)

#clf = AdaBoostClassifier()

clf = lgb.LGBMClassifier(n_jobs=-1)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Learn the digits on the train subset
print("fitting...")
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_valid)

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_valid, predicted)}\n"
)

print(accuracy_score(y_valid, predicted))