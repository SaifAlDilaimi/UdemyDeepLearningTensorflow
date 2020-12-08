# confusion matrix
from confusion_matrix import plot_confusion_matrix

# import sklearn, numpy und keras
from sklearn.metrics import confusion_matrix

import numpy as np
import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

# import matplotlib
import matplotlib.pyplot as plt

# Statische variablen festlegen
epochs = 5
batch_size = 128
learning_rate = 1e-3
momentum = 0.9