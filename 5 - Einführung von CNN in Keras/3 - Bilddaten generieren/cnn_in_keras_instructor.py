# numpy und keras importieren
import numpy as np
import tensorflow as tf
from tensorflow import keras

# import keras layers
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras.layers import MaxPooling2D, Dense, Flatten, Activation
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy

# dataset generieren -> Bilder 1000x100x100x3
# generieren von train, val und test daten
x_train = np.random.random((1000, 100, 100, 3))
x_val = np.random.random((300, 100, 100, 3))
x_test = np.random.random((300, 100, 100, 3))

# generieren von train, val und test labels
y_train = np.random.randint(10, size=(1000, 1))
y_val = np.random.randint(10, size=(300, 1))
y_test = np.random.randint(10, size=(300, 1))

# model labels zu one-hot definieren
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=10)
y_val_one_hot = keras.utils.to_categorical(y_val, num_classes=10)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=10)

# Test ausgabe
print("Shape von x_train: ", x_train.shape)
print("Shape von y_train: ", y_train.shape)
print("Shape von y_train: ", y_train_one_hot.shape)