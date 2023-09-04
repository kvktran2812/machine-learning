import numpy as np
import pandas as pd


def get_series_data_and_label(data, window_size=5):
    X = []
    y = []

    for i in range(len(data) - window_size):
        row = [[a] for a in data[i: i+window_size]]
        X.append(row)
        label = data[i + window_size]
        y.append(label)

    return np.array(X), np.array(y)
