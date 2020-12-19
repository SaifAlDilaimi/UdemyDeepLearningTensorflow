from keras.datasets import cifar10

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Add

from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2

from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

from matplotlib import pyplot as plt

import numpy as np
import os

learning_rate = 1e-3
min_learning_rate = 0.5e-6
batch_size = 128
epochs = 200
n = 3
depth = n * 6 + 2
num_classes = 10
model_name = 'keras_cifar10_trained_model.h5'