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
    cnn_model = Sequential()

    cnn_model.add(Conv2D(filters=32, kernel_size=3, padding="same", input_shape=img_shape))
    cnn_model.add(Activation("relu"))
    cnn_model.add(Conv2D(filters=32, kernel_size=3, padding="same"))
    cnn_model.add(Activation("relu"))
    cnn_model.add(MaxPool2D())

    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=num_classes))
    cnn_model.add(Activation("softmax"))

    cnn_model.summary()

    return cnn_model

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
    data = RandomImageDataset()

    train_dataset = data.get_train_set()
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