'''
XGBOOST on iris flower data set (multiclass classification)

Tutorial from https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7
'''

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn import datasets
import numpy as np


# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# split data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

# transform to DMatrix
D_train = xgb.DMatrix(X_train, label=Y_train)
D_test = xgb.DMatrix(X_test, label=Y_test)

# define xgboost model
param = {
    'eta': 0.3,  # learning rate
    'max_depth': 3,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 3}
epochs = 20

# train
model = xgb.train(param, D_train, epochs)

# predict test set
preds = model.predict(D_test)

# evaluate model and print metrics
best_preds = np.asarray([np.argmax(line) for line in preds])
print("Precision = {}".format(precision_score(Y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(Y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_test, best_preds)))
