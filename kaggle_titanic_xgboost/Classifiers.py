import abc
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np


class Classifier:

    @abc.abstractmethod
    def train(trainset):
        return

    @abc.abstractmethod
    def predict(testset):
        return


class XGClassifier(Classifier):

    def __init__(self, params={
        "eta": 0.2,
        "max_depth": 10,
        "objective": "binary:logistic",
            "eval_metric": "error", "epochs": 20}):

        self.epochs = params['epochs']
        params.pop("epochs")
        self.params = params
        self.model = None

    def train(self, trainset, labels):
        D_train = xgb.DMatrix(trainset, label=labels)
        self.model = xgb.train(self.params, D_train, self.epochs)

    def test(self, testset, labels=None):
        D_val = xgb.DMatrix(testset, label=labels)
        preds = self.model.predict(D_val)
        if labels is not None:
            # print accuracy if we have labels
            best_preds = np.asarray([np.argmax(line) for line in preds])
            print("accuracy: ",
                  metrics.accuracy_score(labels, best_preds))
        return preds


class RFClassifier(Classifier):

    def __init__(self, params={"trees": 100}):
        self.model = RandomForestClassifier(n_estimators=params["trees"])

    def train(self, trainset, labels):
        self.model.fit(trainset, labels)

    def test(self, testset, labels=None):
        preds = self.model.predict(testset)
        if labels is not None:
            print("accuracy: ",
                  metrics.accuracy_score(labels, preds))
        return preds


def classifierFactory(classifier: str):
    if classifier.lower() == "xg":
        return XGClassifier()
    elif classifier.lower() == "rf":
        return RFClassifier()
    else:
        NotImplementedError(classifier + " not implemented")
