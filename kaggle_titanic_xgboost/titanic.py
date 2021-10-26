"""
Apply some standard ML algorithms to the kaggle titanic problem (binary classification).

Currently implemented: XGBOOST, Random Forest

current Kaggle scores: 
    - XGBOOST: 0.77990
    - Random Forest 0.76555
    - AdaBoost: 0.77511

competition:
https://www.kaggle.com/c/titanic/overview
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Classifiers import XGClassifier, RFClassifier, ABClassifier

id = "PassengerId"
label = "Survived"
categorical_cols = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]
# categorical data we want to encode
to_one_hot_cols = ["Sex", "Embarked"]


def get_df(filename, train=True):
    '''
    read data and pre-process into dataframe
    '''
    # read train data
    X = pd.read_csv(filename)

    # fill missing features
    X = X.fillna(method='ffill')

    ids, Y = [], []
    if train:
        Y = X[label]
        X = X.drop([id, label], axis=1)
    else:
        ids = X[id]
        X = X.drop([id], axis=1)

    # one hot encode categorical columns
    onehot_df = X[to_one_hot_cols]
    onehot_df = pd.get_dummies(X[to_one_hot_cols])
    # append one hot encoded columns
    X = pd.concat([X, onehot_df], axis=1)

    return X, Y, ids


def add_features(df):
    """
    do some feature engineering
    """

    # add length of names, as longer names correspond to more important (or more spanish) people
    df["name_len"] = df.apply(lambda row: len(row.Name), axis=1)

    # add label-encoded title (Mr, Mrs, Miss, Dr, ...)
    label_encode = df["Name"]
    label_encode = label_encode.apply(lambda name: name.split(', ')[
        1].split(' ')[0]).to_frame()
    label_encode = label_encode.apply(LabelEncoder().fit_transform)
    df = pd.concat([df, label_encode], axis=1)

    # remove other categorical columns
    df = df.drop(categorical_cols, axis=1)
    return df


# read train data
X_train, Y_train, _ = get_df("data/train.csv", train=True)
X_train = add_features(X_train)

# split into train/validation data
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.2)

classifier = ABClassifier()  # one of: XGClassifier, XGClassifier, RFClassifier
classifier.train(X_train, Y_train)
preds = classifier.test(X_val, Y_val)

# predicting test set and create submission file
with open("submission.txt", 'w') as outfile:
    outfile.write(id + ',' + label + '\n')

    X_test, Y_train, ids = get_df("data/test.csv", train=False)
    X_test = add_features(X_test)

    preds = classifier.test(X_test)
    for prob, id in zip(preds, ids):
        outfile.write(str(id) + ',' + str(int(round(prob))) + '\n')
