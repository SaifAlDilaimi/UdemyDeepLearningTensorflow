import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

from catVSDogDatasetContainer import CATDOGDataset

from PIL import Image

EPOCHS = 5
LEARNING_RATE = 0.0001

LOGS_DIR = os.path.abspath("C:/Users/saifa/Desktop/UdemyDeepLearningTensorflow-main/logs")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)

MODELS_DIR = os.path.abspath("C:/Users/saifa/Desktop/UdemyDeepLearningTensorflow-main/models")
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    catdog_model = Sequential()

    catdog_model.add(Conv2D(filters=24, kernel_size=3, padding="same", input_shape=img_shape))
    catdog_model.add(BatchNormalization())
    catdog_model.add(Activation("relu"))
    catdog_model.add(Conv2D(filters=24, kernel_size=3, padding="same"))
    catdog_model.add(BatchNormalization())
    catdog_model.add(Activation("relu"))
    catdog_model.add(MaxPool2D())

    catdog_model.add(Conv2D(filters=48, kernel_size=3, padding="same"))
    catdog_model.add(BatchNormalization())
    catdog_model.add(Activation("relu"))
    catdog_model.add(Conv2D(filters=48, kernel_size=3, padding="same"))
    catdog_model.add(BatchNormalization())
    catdog_model.add(Activation("relu"))
    catdog_model.add(MaxPool2D())

    catdog_model.add(Conv2D(filters=96, kernel_size=3, padding="same"))
    catdog_model.add(BatchNormalization())
    catdog_model.add(Activation("relu"))
    catdog_model.add(Conv2D(filters=96, kernel_size=3, padding="same"))
    catdog_model.add(BatchNormalization())
    catdog_model.add(Activation("relu"))
    catdog_model.add(MaxPool2D())

    catdog_model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
    catdog_model.add(BatchNormalization())
    catdog_model.add(Activation("relu"))
    catdog_model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
    catdog_model.add(BatchNormalization())
    catdog_model.add(Activation("relu"))
    catdog_model.add(MaxPool2D())

    catdog_model.add(Conv2D(filters=num_classes, kernel_size=3, padding="same"))
    catdog_model.add(BatchNormalization())
    catdog_model.add(GlobalAveragePooling2D())
    catdog_model.add(Activation("softmax"))

    catdog_model.summary()

    return catdog_model

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

def train(model_name: str, data: CATDOGDataset) -> None:
    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()

    model = build_model(data.img_shape, data.num_classes)

    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"]
    )

    model.summary()

    model_log_dir = os.path.join(LOGS_DIR, f"model_{model_name}")

    tb_callback = TensorBoard(
        log_dir=model_log_dir,
        histogram_freq=0,
        profile_batch=0,
        write_graph=False
    )

    mc_callback = ModelCheckpoint(
        os.path.join(MODELS_DIR, model_name),
        monitor='val_loss',
        mode='auto',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    es_callback = EarlyStopping(
        monitor='val_loss', # auch val_accuracy probieren
        mode='auto',
        patience=5,
        verbose=1
    )

    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_accuracy', # auch val_accuracy probieren
        mode='auto',
        factor=0.9,
        patience=3,
        cooldown=0,
        min_lr=0,
        verbose=1
    )

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        batch_size=data.batch_size,
        verbose=1,
        validation_data=val_dataset,
        callbacks=[tb_callback, mc_callback, reduce_lr_callback]
    )

def evaluate(model_name: str, data: CATDOGDataset) -> None:
    model_path = os.path.join(MODELS_DIR, model_name)

    model = load_model(model_path)

    test_dataset = data.get_test_set()

    y_pred = model.predict(test_dataset)
    plot_predicted_images(y_pred, test_dataset)

    score = model.evaluate(test_dataset)
    print(f'The Accuracy on the Test set is {score[1]:.2%} and the loss is {score[0]:.3f}')

if __name__ == "__main__":
    data = CATDOGDataset()

    model_name = f"catvsdog_4block_load_model"

    if not os.path.exists(os.path.join(MODELS_DIR, model_name)):
        train(model_name, data)

    evaluate(model_name, data)
