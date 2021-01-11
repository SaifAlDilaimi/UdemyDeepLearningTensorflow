import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from typing import Tuple

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import add

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

from cifar10DatasetContainer import CIFAR10Dataset

EPOCHS = 125
LEARNING_RATE = 0.001

LOGS_DIR = os.path.abspath("C:/Users/saifa/Desktop/UdemyDeepLearningTensorflow-main/logs")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)

MODELS_DIR = os.path.abspath("C:/Users/saifa/Desktop/UdemyDeepLearningTensorflow-main/models")
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Model:
    inputL = Input(shape=img_shape)

    x = Conv2D(filters=64, kernel_size=3, padding="same")(inputL)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D()(x)

    residual = Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)
    residual = BatchNormalization()(residual)

    x = Conv2D(filters=128, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv2D(filters=128, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D()(x)

    x = add([x, residual])

    residual = Conv2D(filters=256, kernel_size=3, strides=2, padding="same")(x)
    residual = BatchNormalization()(residual)

    x = Conv2D(filters=256, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv2D(filters=256, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D()(x)

    x = add([x, residual])

    x = Conv2D(filters=num_classes, kernel_size=3, padding="same")(x)
    x = GlobalAveragePooling2D()(x)
    prediction_layer = Activation("softmax")(x)

    cifar_model = Model(inputs=[inputL], outputs=[prediction_layer])
    cifar_model.summary()

    return cifar_model

def plot_predicted_images(y_pred: np.ndarray, test_dataset: tf.data.Dataset) -> None:
    batch = test_dataset.take(1)
    images, labels = batch.as_numpy_iterator().next()
    for i in range(10):
        high_prob_class = np.argmax(y_pred[i])
        prob = y_pred[i][high_prob_class]
        print(f'I am {prob:.2%} sure I am Class {high_prob_class}. The true class is {np.argmax(labels[i])}')
        plt.imshow(images[i])
        plt.show()

def train(model_name: str, data: CIFAR10Dataset) -> None:
    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()

    model = build_model(data.img_shape, data.num_classes)

    model.compile(
        loss=categorical_crossentropy,
        optimizer=RMSprop(learning_rate=LEARNING_RATE),
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
        monitor='val_accuracy',
        mode='auto',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        mode='auto',
        factor=0.9,
        patience=3,
        cooldown=0,
        min_lr=0,
        verbose=1
    )

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        batch_size=data.batch_size,
        verbose=1,
        validation_data=val_dataset,
        callbacks=[tb_callback, mc_callback, reduce_lr_callback]
    )

def evaluate(model_name: str, data: CIFAR10Dataset) -> None:
    model_path = os.path.join(MODELS_DIR, model_name)

    model = load_model(model_path)

    test_dataset = data.get_test_set()

    y_pred = model.predict(test_dataset)
    plot_predicted_images(y_pred, test_dataset)

    score = model.evaluate(test_dataset)
    print(f'The Accuracy on the Test set is {score[1]:.2%} and the loss is {score[0]:.3f}')

if __name__ == "__main__":
    data = CIFAR10Dataset()

    model_name = f"cifar10_residuals_RDLR"

    if not os.path.exists(os.path.join(MODELS_DIR, model_name)):
        train(model_name, data)

    evaluate(model_name, data)
