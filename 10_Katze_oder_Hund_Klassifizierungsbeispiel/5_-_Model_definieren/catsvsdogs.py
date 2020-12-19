# keras imports vorbereiten
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization, Flatten, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import RMSprop

# Statische variablen setzen
img_width, img_height = 150, 150
working_dir = os.path.join(os.path.curdir, "images")
train_data_dir = os.path.join(working_dir, "train")
validate_data_dir = os.path.join(working_dir, "validation")
test_data_dir = os.path.join(working_dir, "test")
nb_train_samples = 5000
nb_validation_samples = 1000

# Hyperparameter
epochs = 10
batch_size = 16
learning_rate = 0.0001 # 0.0001
input_shape = (img_width, img_height, 3)

# model definieren
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(2))
model.add(Activation('softmax'))