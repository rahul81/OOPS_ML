import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import pickle
import json


class ML:
    def __init__(
        self,
    ):

        self.grid_searchCV = None
        self.clf_and_params = list()

        self.make_clf_parmas()

    def make_clf_parmas(self):

        clf = KNeighborsClassifier()
        params = {
            "n_neighbors": [5, 7, 9, 11, 13, 15],
            "leaf_size": [1, 2, 3, 5],
            "weights": ["uniform", "distance"],
        }
        self.clf_and_params.append((clf, params))

        clf = LogisticRegression()
        params = {"penalty": ["l1", "l2"], "C": np.logspace(0, 4, 10)}
        self.clf_and_params.append((clf, params))

        clf = DecisionTreeClassifier()
        params = {
            "max_features": ["auto", "sqrt", "log2"],
            "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "min_samples_leaf": [1],
            "random_state": [123],
        }

        self.clf_and_params.append((clf, params))

        clf = RandomForestClassifier()
        params = {
            "n_estimators": [4, 6, 9],
            "max_features": ["log2", "sqrt", "auto"],
            "criterion": ["entropy", "gini"],
            "max_depth": [2, 3, 5, 10],
            "min_samples_split": [2, 3, 5],
            "min_samples_leaf": [1, 5, 8],
        }

        self.clf_and_params.append((clf, params))

        clf = MultinomialNB()
        params = {"alpha": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2]}

        self.clf_and_params.append((clf, params))

    def grid_search(self, X_train, X_test, y_train, y_test, n_splits):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        clf_and_params = self.clf_and_params
        models = []
        best_params = {}
        self.results = {}
        for clf, params in clf_and_params:
            self.current_clf_name = clf.__class__.__name__
            print("Traning model : ", self.current_clf_name)
            grid_search_clf = GridSearchCV(
                clf, params, cv=self.cv, scoring=make_scorer(roc_auc_score)
            )
            grid_search_clf.fit(self.X_train, self.y_train)
            self.Y_pred = grid_search_clf.predict(self.X_test)
            print(
                f"Traning score on model : {self.current_clf_name} : {grid_search_clf.score(X_train,y_train)}"
            )
            best_params[self.current_clf_name] = grid_search_clf.best_params_
            score = roc_auc_score(self.y_test, self.Y_pred)
            print(f"Test ROC AUC score on model : {self.current_clf_name} : {score}")
            pickle.dump(
                self.current_clf_name, open(f"models/{self.current_clf_name}", "wb")
            )

        with open("best_params/best_params.json", "w") as f:
            json.dump(best_params, f)
            f.close()

    def model_fit(self, model, X_train, y_train, x_test, y_test):

        model.fit(X_train, y_train)

        preds = model.predict(X_train)

        print("================================================")
        print(
            f"Train roc auc score for model {model.__class__.__name__} : {roc_auc_score(y_train,preds)}"
        )

        preds = model.predict(x_test)
        print(
            f"Test roc auc score for model {model.__class__.__name__} : {roc_auc_score(y_test,preds)}"
        )
        print("================================================")

        print(f"Saving model {model.__class__.__name__}...")

        pickle.dump(model, open(f"models/{model.__class__.__name__}.pkl", "wb"))

        return model

    def model_inference(self, model, features, targets):

        preds = model.predict(features)

        if targets is not None:

            print(
                "Test ROC AUC score on model {} : {}".format(
                    model.__class__.__name__, roc_auc_score(targets, preds)
                )
            )

        else:

            print("Returning predictions no targets provided to score")

        return preds
