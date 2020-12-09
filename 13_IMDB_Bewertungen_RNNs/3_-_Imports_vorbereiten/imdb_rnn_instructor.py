# keras und diverses importieren
import os

import keras
import numpy as np
# dataset
from keras.datasets import imdb
# layers
from keras.layers import Input, Activation, Conv1D, MaxPool1D
from keras.layers import Embedding, LSTM, CuDNNLSTM, Dense              
# optimizer + loss
from keras.losses import binary_crossentropy
from keras.optimizers import Adagrad, Adam, RMSprop
# keras functional api
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence

# statische variablen
epochs = 3
learning_rate = 1e-3
batch_size = 64 # besser für weight update
vocab_size = 100000
embedding_length = 32
max_review_length = 500

def prepare_dataset():
    # dataset vorbereiten
    pass


def train(x_train, y_train, x_test, y_test, x_val, y_val):
    # model definieren
    pass

def predict():
    # reviews zum vorhersagen
    pass

def main():
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = prepare_dataset()
    
    if not os.path.exists("review-predictor.npz"):
        train(x_train, y_train, x_test, y_test, x_val, y_val)
    
    predict()

if __name__ == '__main__':
    main()
