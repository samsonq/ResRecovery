import os
import sys
import numpy as np
from scipy import stats
import pandas as pd
import pickle
import io
from io import StringIO
from flask import Flask, render_template, request, jsonify, make_response
from aggregate import *
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
baseline_model = pickle.load(open("./model/baseline_model.pkl", "rb"))
extended_model = pickle.load(open("./model/final_model.pkl", "rb"))
chunk_size = 120


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
    X = create_features([df], chunk_size)
    X = X[['dwl_bytes_avg', 'dwl_peak_prom', 'upl_bytes_std', 'dwl_bytes_std', 'dwl_max_psd', 'dwl_num_peak']]
    array_preds = extended_model.predict(X)
    prediction = stats.mode(array_preds)[0][0]
    resolutions = {1: "240p", 2: "480p", 3: "1080p"}
    array_preds = [resolutions[i] for i in array_preds]
    display_text = "The predicted resolutions for each interval are: {} \n Overall, " \
                   "the most commonly predicted resolution is: {}.".format(array_preds, resolutions[prediction])

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
    X = create_features([df], chunk_size)
    X = X[['dwl_bytes_avg', 'dwl_peak_prom', 'upl_bytes_std', 'dwl_bytes_std', 'dwl_max_psd', 'dwl_num_peak']]
    array_preds = extended_model.predict(X)
    prediction = stats.mode(array_preds)[0][0]
    resolutions = {1: "Low", 2: "Medium", 3: "High"}
    array_preds = [resolutions[i] for i in array_preds]
    display_text = "The predicted resolutions for each interval are: {} \n Overall, " \
                   "the most commonly predicted resolution is: {}.".format(array_preds, resolutions[prediction])

    return render_template('index.html', prediction_text_extended=display_text)


if __name__ == '__main__':
    app.run(debug=True)
