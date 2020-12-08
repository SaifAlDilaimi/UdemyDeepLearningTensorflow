import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.layers import LSTM, Dense, Input
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
    x_train = np.load("x_train.npy")
    x_test = np.load("x_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")


    input_layer = Input(shape=(7, 1))
    x = LSTM(256)(input_layer) 
    prediction = Dense(1)(x)

    model = Model(inputs=[input_layer], outputs=[prediction])

    opt = Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss=mean_squared_error)

    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(loss)
    plt.plot(val_loss)
    plt.show()

    model.save("stock_predictor")

def evaluate():
    model = load_model("stock_predictor")

    x_train = np.load("x_train.npy")
    x_test = np.load("x_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")

    data = pd.read_csv('all_stocks_5yr.csv')
    cl = data[data['Name'] == 'MMM'].close
    scaler = MinMaxScaler()
    close = cl.values.reshape(cl.shape[0], 1)
    close = scaler.fit_transform(close)
    
    predictions = model.predict(x_test)

    print(x_test.shape)
    
    x_test = scaler.inverse_transform(x_test.reshape(-1, 7))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions = scaler.inverse_transform(predictions)


    plt.title("Evaluate")   #|steht fuer Anzahl Listeintraege
    plt.plot(y_test, color="blue", label="OriginalKurs")
    plt.plot(predictions, color="orange", label="Vorhersage")
    plt.show()


def main():
    if not os.path.exists("x_train.npy"):
        prepare_dataset()
    if not os.path.exists("stock_predictor"):
        train()
    
    evaluate()


if __name__ == "__main__":
    main()