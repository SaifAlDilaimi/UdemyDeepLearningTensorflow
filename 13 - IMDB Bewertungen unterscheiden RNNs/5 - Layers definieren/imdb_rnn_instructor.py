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
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    # pad reviews zur gleichen länge
    x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

    # testset teilen für validation set
    x_size_test = int(len(x_test) * 0.66)
    x_val = x_test[x_size_test:]
    y_val = y_test[x_size_test:]

    x_test = x_test[:x_size_test]
    y_test = y_test[:x_size_test]

    print("Dataset:")
    print("Training: ", x_train.shape, y_train.shape)
    print("Test: ", x_test.shape, y_test.shape)
    print("Validation: ", x_val.shape, y_val.shape)

    return (x_train, y_train), (x_test, y_test), (x_val, y_val)


def train(x_train, y_train, x_test, y_test, x_val, y_val):
    # model definieren
    review_input = Input(shape=(max_review_length,), name="review_input")
    # layers
    x = Embedding(vocab_size, embedding_length, input_length=max_review_length)(review_input)
    x = Conv1D(filters=32, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=2)(x)
    x = LSTM(100)(x)
    x = Dense(1)(x)
    output = Activation('sigmoid')(x)   

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
