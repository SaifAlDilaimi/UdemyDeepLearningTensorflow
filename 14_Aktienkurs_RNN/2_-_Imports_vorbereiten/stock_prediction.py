import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.layers import LSTM, Dense, CuDNNLSTM, Input
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

learning_rate = 1e-3
look_back = 7
batch_size = 32
epochs = 300

def prepare_dataset():
    pass

def train():
    pass

def evaluate():
    pass

def main():
    if not os.path.exists("x_train.npy"):
        prepare_dataset()
    if not os.path.exists("stock_predictor"):
        train()
    
    evaluate()


if __name__ == "__main__":
    main()