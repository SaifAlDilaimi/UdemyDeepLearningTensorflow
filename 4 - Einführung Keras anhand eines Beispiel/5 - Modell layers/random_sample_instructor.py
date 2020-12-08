# importieren von keras und numpy
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical

# dataset vorbereiten
# generieren von train, val und test daten
x_train = np.random.random((1000, 100))
x_val = np.random.random((100, 100))
x_test = np.random.random((100, 100))

# generieren von train, val und test labels
y_train = np.random.randint(0, 10, size=(1000, 1))
y_val = np.random.randint(0, 10, size=(100, 1))
y_test = np.random.randint(0, 10, size=(100, 1))

# Test ausgabe
print(f"Shape von x_train: {x_train.shape}")
print(f"Shape von y_train: {y_train.shape}")

# model definieren
model = Sequential()

# model layers definieren
model.add(Dense(32, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))