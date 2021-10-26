import abc
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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
    """
    XGBoost classifier.
    """

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
    """
    Random Forest classifier.
    """

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


class ABClassifier(Classifier):
    """
    AdaBoost classifier.
    """

    def __init__(self, params={"max_depth": 3, "random_state": 0, "n_estimators": 3, "algorithm": "SAMME"}):
        base_estimator = DecisionTreeClassifier(
            max_depth=params["max_depth"], random_state=params["random_state"])
        self.model = AdaBoostClassifier(base_estimator=base_estimator,
                                        n_estimators=params["n_estimators"], algorithm=params["algorithm"],
                                        random_state=params["random_state"])

    def train(self, trainset, labels):
        self.model.fit(trainset.values, labels.values)

    def test(self, testset, labels=None):
        preds = self.model.predict(testset.values)
        if labels is not None:
            print("accuracy: ",
                  metrics.accuracy_score(labels, preds))
        return preds
