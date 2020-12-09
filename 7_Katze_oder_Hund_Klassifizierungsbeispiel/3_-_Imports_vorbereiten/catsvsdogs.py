# keras imports vorbereiten
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization, Flatten, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import RMSprop