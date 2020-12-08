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
