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

def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    pass

def plot_history(history: dict) -> None:
    pass


if __name__ == "__main__":
    pass