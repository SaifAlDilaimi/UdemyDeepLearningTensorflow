import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

from mnistDatasetContainer import MNISTDataset
from confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

EPOCHS = 10
LEARNING_RATE = 0.001

def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    mnist_model = Sequential()

    mnist_model.add(Conv2D(filters=32, kernel_size=3, padding="same", input_shape=img_shape))
    mnist_model.add(Activation("relu"))
    mnist_model.add(Conv2D(filters=32, kernel_size=3, padding="same"))
    mnist_model.add(Activation("relu"))
    mnist_model.add(MaxPool2D())

    mnist_model.add(Flatten())
    mnist_model.add(Dense(128))
    mnist_model.add(Dropout(0.2))
    mnist_model.add(Dense(units=num_classes))
    mnist_model.add(Activation("softmax"))

    mnist_model.summary()

    return mnist_model

def plot_history(history: dict) -> None:
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    pass