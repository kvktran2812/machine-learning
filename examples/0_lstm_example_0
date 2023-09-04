import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os

def get_data_to_train(data, window_size=5):
    X = []
    y = []

    for i in range(len(data)-window_size):
        row = [[a] for a in data[i:i+window_size]]
        X.append(row)
        label = data[i+window_size]
        y.append(label)

    return np.array(X), np.array(y)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = tf.keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip", extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)
data = df['Tdew (degC)'].to_numpy()
X, y = get_data_to_train(data)

model = Sequential()
model.add(InputLayer((5, 1)))
model.add(LSTM(64))
model.add(Dense(8, 'relu'))
model.add(Dense(8, 'relu'))
model.add(Dense(1, 'linear'))
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001))
model.fit(X_train, y_train, epochs=10)

model.evaluate(X_test, y_test)
