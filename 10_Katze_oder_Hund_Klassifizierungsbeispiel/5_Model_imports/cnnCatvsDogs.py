import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

# ...

def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    pass

def plot_history(history: dict) -> None:
	# plot loss
    plt.subplot(121)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.grid()
    # plot accuracy
    plt.subplot(122)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    pass