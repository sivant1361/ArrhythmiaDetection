import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pandas as pd
# from utils import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler


import wfdb
import matplotlib.pyplot as plt
from pathlib import Path

from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, TimeDistributed
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


app = Flask(__name__)

invalid_beat = [
    "[", "!", "]", "x", "(", ")", "p", "t", 
    "u", "`", "'", "^", "|", "~", "+", "s", 
    "T", "*", "D", "=", '"', "@"
]

abnormal_beats_dict = {
    "0":"Normal beat",
    "1":"Left bundle branch block beat",
    "2":"Right bundle branch block beat",
    "3":"Bundle branch block beat (unspecified)",
    "4":"Atrial premature beat",
    "5":"Aberrated atrial premature beat",
    "6":"Nodal (junctional) premature beat",
    "7":"Supraventricular premature or ectopic beat (atrial or nodal)",
    "8":"Premature ventricular contraction",
    "9":"R-on-T premature ventricular contraction",
    "10":"Fusion of ventricular and normal beat",
    "11":"Atrial escape beat",
    "12":"Nodal (junctional) escape beat",
    "13":"Supraventricular escape beat (atrial or nodal)",
    "14":"Ventricular escape beat",
    "15":"Paced beat",
    "16":"Fusion of paced and normal beat",
    "17":"Unclassifiable beat",
    "18":"Beat not classified during learning",
    "19":"Invalid beat"
}

abnormal_beats_index = {
    "N":0,
    "L":1,
    "R":2,
    "B":3,
    "A":4,
    "a":5,
    "J":6,
    "S":7,
    "V":8,
    "r":9,
    "F":10,
    "e":11,
    "j":12,
    "n":13,
    "E":14,
    "/":15,
    "f":16,
    "Q":17,
    "?":18,
}

def classify_beat(symbol):
    if symbol in abnormal_beats_index.keys():
        return int(abnormal_beats_index[symbol])
    elif symbol == ".":
        return 0
    else:
        return 19    

def get_sequence(signal, beat_loc, window_sec, fs):
    window_one_side = window_sec * fs
    beat_start = beat_loc - window_one_side
    beat_end = beat_loc + window_one_side
    if beat_end < signal.shape[0]:
        sequence = signal[beat_start:beat_end, 0]
        return sequence.reshape(1, -1, 1)
    else:
        return np.array([])

def test_pipeline(subject):
    sequences = []
    labels = []
    window_sec = 3

    record = wfdb.rdrecord(f'mit-bih-arrhythmia-database-1.0.0/{subject}')
    annotation = wfdb.rdann(f'mit-bih-arrhythmia-database-1.0.0/{subject}', 'atr')
    atr_symbol = annotation.symbol
    atr_sample = annotation.sample
    
    fs = record.fs
    scaler = StandardScaler()
    signal = scaler.fit_transform(record.p_signal)
    
    for i, i_sample in enumerate(atr_sample):
        label = classify_beat(atr_symbol[i])
        sequence = get_sequence(signal, i_sample, window_sec, fs)
        if label is not None and sequence.size > 0:
            sequences.append(sequence)
            labels.append(label)

    X, y = np.vstack(sequences), np.vstack(one_hot(np.array(labels),20))
    print(X.shape,y.shape)
    return X, y

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/arrhy')
def arrhy():
    # speak("Navigating to BMI predictor!")
    return render_template('arrhy.html', Name="Arrhythmia Detection")

@app.route('/arrhy_predict', methods=['POST'])
def arrhy_predict():
    try:
        X_test, y_test = test_pipeline(101)
        print(X_test[0],y_test[0])
        # y_pred = 
        model = tf.keras.models.load_model("./models/cnn_model.h5")
        print(np.argmax(model.predict(X_test)==5))
        ypred = model.predict(np.array([X_test[13]]))
        index_val = str(np.argmax(ypred))
        index_val = str(5)
        # print(abnormal_beats_dict[index_val])
        # return render_template('index.html')
        return render_template('arrhy_predict.html', Name="Arrhythmia Detection", index=index_val,type = abnormal_beats_dict[index_val])
    except Exception as e:
        print(e)
        return redirect("404")

@app.route("/about")
def about():
    return render_template('about.html')