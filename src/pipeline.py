import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from info import Information
from preprocess import Preprocess, make_features
from ml import ML
import pickle


class Sentiment_Analysis:
    def __init__(self, max_features=5000, test=False, feature_strategy="Tfidf"):
        self.info = Information()
        self.preprocess = Preprocess()
        self.make_features = make_features(test)
        self.ml = ML()
        self.resue = test

        if self.resue:
            self.vectorizer = pickle.load(open("vectorizers/vectorizer.pkl", "rb"))
            print("INFERENCE PHASE")
        else:
            print("TRAINING PHASE")
            if feature_strategy == "Tfidf":
                self.vectorizer = TfidfVectorizer(max_features=max_features)
            elif feature_strategy == "Count":
                self.vectorizer = CountVectorizer(max_features=max_features)

    def transform(self, data, test=False):

        if test == False:
            self.info.info(data)
        data = self.preprocess.clean(data, emoticon_pattern=1)
        features = self.make_features.get_features(data["clean_text"], self.vectorizer)

        print("New features shape:", features.shape)

        if test:
            return features
        else:
            return features, data["target"]

    def get_splits(self, features, targets, test_size):

        print("Getting train and test splits...")
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(
            features, targets, test_size=test_size, stratify=targets, random_state=42
        )

        return Xtrain, Xtest, Ytrain, Ytest

    def train_model(
        self, df, model, sample_data=True, sampling_ratio=(0.4, 0.7), test_size=0.2
    ):

        features, targets = self.transform(df)
        Xtrain, Xtest, Ytrain, Ytest = self.get_splits(features, targets, test_size)

        if sample_data:
            Xtrain, Ytrain = self.make_features.sample_data(
                Xtrain, Ytrain, sampling_ratio[0], sampling_ratio[1]
            )

        model = self.ml.model_fit(model, Xtrain, Ytrain, Xtest, Ytest)

        return model

    def train_gridsearch(
        self, df, sample_data=True, sampling_ratio=(0.4, 0.7), n_splits=5, test_size=0.2
    ):

        features, targets = self.transform(df)
        Xtrain, Xtest, Ytrain, Ytest = self.get_splits(features, targets, test_size)

        Xtrain, Ytrain = self.make_features.sample_data(
            Xtrain, Ytrain, sampling_ratio[0], sampling_ratio[1]
        )

        self.ml.grid_search(Xtrain, Xtest, Ytrain, Ytest, n_splits=n_splits)

    def model_inference(self, model, df, targets=None):

        features = self.transform(df, test=True)
        preds = self.ml.model_inference(model, features, targets=targets)

        return preds
