import matplotlib.pyplot as plt

from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

from datasetContainer import RandomImageDataset

EPOCHS = 10
LEARNING_RATE = 0.0001
