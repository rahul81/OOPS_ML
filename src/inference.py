import pandas as pd 
import numpy as np 

import pickle

import pipeline

if __name__ == "__main__":
    df = pd.read_csv("../data/airline_sentiment_analysis.csv")
    df['airline_sentiment'] = df['airline_sentiment'].str.lower()
    df['target'] = df['airline_sentiment'].map({"positive":1,"negative":0})
    df.drop(["Unnamed: 0","airline_sentiment"],axis=1,inplace=True)


    model = pickle.load(open("models/MultinomialNB.pkl","rb"))

    SA = pipeline.Sentiment_Analysis(test=True)
    preds = SA.model_inference(model,df,targets=df['target'])

