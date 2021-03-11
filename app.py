import os
import sys
import numpy as np
from scipy import stats
import pandas as pd
import pickle
import io
from io import StringIO
from flask import Flask, render_template, request, jsonify, make_response
from features import create_features, features
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
baseline_model = pickle.load(open("./model/baseline_model.pkl", "rb"))
extended_model = pickle.load(open("./model/final_model.pkl", "rb"))
chunk_size = 300


@app.route('/')
def home():
    return render_template('index.html')


def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/baseline', methods=["POST"])
def predict_baseline():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    stream.seek(0)
    result = transform(stream.read())
    df = pd.read_csv(StringIO(result))

    # Preprocessing & Feature Building
    X = create_features(df, chunk_size, 5, 1, .5)
    X = pd.DataFrame(columns=features, data=X)
    array_preds = baseline_model.predict(X)
    prediction = stats.mode(array_preds)[0][0]
    display_text = "The predicted resolutions for each interval are: {} \n Overall, " \
                   "the most commonly predicted resolution is: {}.".format(array_preds, prediction)

    return render_template('index.html', prediction_text_extended=display_text)


@app.route('/extended', methods=["POST"])
def predict_extended():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    stream.seek(0)
    result = transform(stream.read())
    df = pd.read_csv(StringIO(result))

    # Preprocessing & Feature Building
    X = create_features(df, chunk_size, 5, 1, .5)
    X = pd.DataFrame(columns=features, data=X)
    array_preds = extended_model.predict(X)
    prediction = stats.mode(array_preds)[0][0]
    display_text = "The predicted resolutions for each interval are: {} \n Overall, " \
                   "the most commonly predicted resolution is: {}.".format(array_preds, prediction)

    return render_template('index.html', prediction_text_extended=display_text)


if __name__ == '__main__':
    app.run(debug=True)
