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

from PIL import Image

EPOCHS = 3
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

def plot_matrix(y_pred: np.ndarray, dataset: tf.data.Dataset) -> None:
    y_pred = np.argmax(y_pred, axis=1)

    y_test = np.concatenate([y for x, y in dataset], axis=0)
    y_test = np.argmax(y_test, axis=1)

    cnf_matrix = confusion_matrix(y_test, y_pred)
    class_names = [str(cl) for cl in range(10)]

    plt.figure()
    plot_confusion_matrix(
        cnf_matrix,
        classes=class_names
    )
    plt.show()


if __name__ == "__main__":
    data = MNISTDataset()

    train_dataset = data.get_train_set()
    test_dataset = data.get_test_set()
    val_dataset = data.get_val_set()

    model = build_model(data.img_shape, data.num_classes)

    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"]
    )

    model.summary()

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        batch_size=data.batch_size,
        verbose=1,
        validation_data=val_dataset
    )

    plot_history(history)

    # Plot confusion
    y_pred = model.predict(test_dataset)
    plot_matrix(y_pred, test_dataset)

    # Load custom image
    x = Image.open('2.jpg')
    print(x.format)
    print(x.size)
    print(x.mode)

    x = x.convert('L')
    print(x.mode)

    x = np.array(x, dtype=np.float32)
    x -= 255
    x /= 255
    x = x.reshape(1, 28, 28, 1)
    print(x.shape)
    x_pred = model.predict(x)
    print(f'Custom image class prediction: {np.argmax(x_pred)}')
    plt.imshow(x[0], cmap='Greys')
    plt.show()