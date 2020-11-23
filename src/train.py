import pandas as pd
import numpy as np 

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
import pipeline



if __name__ == "__main__":
    df = pd.read_csv("../data/airline_sentiment_analysis.csv")
    df['airline_sentiment'] = df['airline_sentiment'].str.lower()
    df['target'] = df['airline_sentiment'].map({"positive":1,"negative":0})
    df.drop(["Unnamed: 0","airline_sentiment"],axis=1,inplace=True)

    # for a in [.2,.3,.4,.5,.6,.8,.9,1,1.5,2]:
    #     model = MultinomialNB(alpha=a)
    #     SA = pipeline.Sentiment_Analysis()
    #     SA.train_model(df,model)
    #     print("**********************")
    #     print("Model with Alpha = ",a)
    #     print("**********************")

    for a in [10,25,50,100,200,300,400,500]:
        model = BaggingClassifier(MultinomialNB(alpha=.6), n_estimators=a)
        SA = pipeline.Sentiment_Analysis()
        SA.train_model(df,model)
        print("**********************")
        print("Model with n_estimators = ",a)
        print("**********************")




