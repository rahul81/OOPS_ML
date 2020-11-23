import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
import pipeline


if __name__ == "__main__":
    df = pd.read_csv("../data/airline_sentiment_analysis.csv")
    df["airline_sentiment"] = df["airline_sentiment"].str.lower()
    df["target"] = df["airline_sentiment"].map({"positive": 1, "negative": 0})
    df.drop(["Unnamed: 0", "airline_sentiment"], axis=1, inplace=True)

    for a in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1, 1.5, 2]:
        model = MultinomialNB(alpha=a)
        SA = pipeline.Sentiment_Analysis()
        SA.train_model(df, model)
        print("**********************")
        print("Model with Alpha = ", a)
        print("**********************")
