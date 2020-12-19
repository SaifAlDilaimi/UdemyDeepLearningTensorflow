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
    data = pd.read_csv('all_stocks_5yr.csv')
    cl = data[data['Name'] == 'MMM'].close
    print(cl[:5])
    scaler = MinMaxScaler()
    print(cl.shape)
    close = cl.values.reshape(cl.shape[0], 1)
    print(close.shape)
    close = scaler.fit_transform(close)
    print(close)

    # daten aufteilen in sequenzen mit 7 werten (letzten 7 tage)
    x, y = [], []
    for i in range(len(close)-look_back-1):
        x.append(close[i:(i+look_back), 0])
        y.append(close[(i+look_back), 0])
    x, y = np.array(x), np.array(y)

    # 80% train, 20% test
    x_train, x_test = x[:int(x.shape[0] * 0.80)], x[int(x.shape[0] * 0.80):]
    y_train, y_test = y[:int(y.shape[0] * 0.80)], y[int(y.shape[0] * 0.80):]
        
    #Reshape data for (Sample,Timestep,Features) 
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    np.save("x_train", x_train)
    np.save("y_train", y_train)
    np.save("x_test", x_test)
    np.save("y_test", y_test)


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