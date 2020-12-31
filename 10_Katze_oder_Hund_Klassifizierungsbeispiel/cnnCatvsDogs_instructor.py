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

from catVSDogDatasetContainer import CATDOGDataset

from PIL import Image

EPOCHS = 10
LEARNING_RATE = 0.0001

def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    catdog_model = Sequential()

    catdog_model.add(Conv2D(filters=24, kernel_size=3, padding="same", input_shape=img_shape))
    catdog_model.add(Activation("relu"))
    catdog_model.add(Conv2D(filters=24, kernel_size=3, padding="same"))
    catdog_model.add(Activation("relu"))
    catdog_model.add(MaxPool2D())

    catdog_model.add(Flatten())
    catdog_model.add(Dense(128))
    catdog_model.add(Dense(units=num_classes))
    catdog_model.add(Activation("softmax"))

    catdog_model.summary()

    return catdog_model

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

def plot_predicted_images(y_pred: np.ndarray, test_dataset: tf.data.Dataset) -> None:
    batch = test_dataset.take(1)
    images, _ = batch.as_numpy_iterator().next()
    for i in range(10):
        if y_pred[i][0] > 0.5:
            print("I am {a:.2%} sure I am Cat".format(a=y_pred[i][0]))
        else:
            print("I am {a:.2%} sure I am Dog".format(a=(1-y_pred[i][0])))
        plt.imshow(images[i])
        plt.show()

if __name__ == "__main__":
    data = CATDOGDataset()

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

    y_pred = model.predict(test_dataset)
    plot_predicted_images(y_pred, test_dataset)

    score = model.evaluate(test_dataset)
    print(f'The Accuracy on the Test set is {score[1]:.2%} and the loss is {score[0]:.3f}')