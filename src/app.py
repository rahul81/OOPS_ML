#importing required packages
from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.view import view_config

import pipeline

import pandas as pd
import pickle


@view_config(route_name="hello")
def hello_world(request):

    return Response("Server is Up and running")


@view_config(route_name="check", renderer="json", request_method="POST", openapi=True)
def check(request):
    text = request.openapi_validated.body["text"]
    df = pd.DataFrame(pd.Series(text), columns=["text"])
    pred = SA.model_inference(model, df)

    if pred[0] == 0:
        return {"Prediction": "Negative Sentiment"}
    return {"Prediction": "Positive Sentiment"}


if __name__ == "__main__":

    model = pickle.load(open("models/MultinomialNB.pkl", "rb"))
    SA = pipeline.Sentiment_Analysis(test=True)

    with Configurator() as config:
        config.include("pyramid_openapi3")
        config.pyramid_openapi3_spec("docs/apidocs.yaml")
        config.pyramid_openapi3_add_explorer()
        config.add_route("check", "/check")
        config.add_route("hello", "/")
        config.scan(".")
        app = config.make_wsgi_app()
    print("Server started at http://localhost:5000")
    print("Swagger UI documentation can be found at http://localhost:5000/docs/")
    server = make_server("0.0.0.0", 5000, app)
    server.serve_forever()
