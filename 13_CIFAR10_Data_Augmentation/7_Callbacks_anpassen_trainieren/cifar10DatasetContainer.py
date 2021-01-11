import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomTranslation
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.datasets import cifar10

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

np.random.seed(0)
tf.random.set_seed(0)

IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

class CIFAR10Dataset():
    def __init__(self, test_size: float = 0.2, validation_size: float = 0.10) -> None:
        # User-definen constants
        self.num_classes = 10
        self.batch_size = 128
        
        # dataset laden
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Split the dataset into val
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size)

        # Preprocess x data
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)
        self.x_val = x_val.astype(np.float32)

        # Preprocess y data
        self.y_train = to_categorical(y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(y_test, num_classes=self.num_classes)
        self.y_val = to_categorical(y_val, num_classes=self.num_classes)

        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.val_size = self.x_val.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.channels = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.channels)

        # Expand training dataset with augmented images
        self.data_augmentation()

        # tf.data Datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        self.train_dataset = self._prepare_dataset(self.train_dataset, shuffle=True, augment=True)
        self.test_dataset = self._prepare_dataset(self.test_dataset)
        self.val_dataset = self._prepare_dataset(self.val_dataset)

    def get_train_set(self) -> tf.data.Dataset:
        return self.train_dataset

    def get_test_set(self) -> tf.data.Dataset:
        return self.test_dataset

    def get_val_set(self) -> tf.data.Dataset:
        return self.val_dataset

    def data_augmentation(self, augment_size: int = 5_000) -> None:
        image_generator = ImageDataGenerator(
            rotation_range=25,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2
        )
        # Fit the data generator
        image_generator.fit(self.x_train, augment=True)
        # Get random train images for the data augmentation
        rand_idxs = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[rand_idxs].copy()
        y_augmented = self.y_train[rand_idxs].copy()
        x_augmented = image_generator.flow(
            x_augmented,
            np.zeros(augment_size),
            batch_size=augment_size,
            shuffle=False
        ).next()[0]
        for x_aug, x in zip(x_augmented[:5], self.x_train[rand_idxs][:5]):
            plt.subplot(121)
            plt.title('Augmented')
            plt.imshow(x_aug / 255.0)
            plt.subplot(122)
            plt.title('Original')
            plt.imshow(x / 255.0)
            plt.show()
        # Append the augmented images to the train set
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]

    @staticmethod
    def _build_z_score(
        x_train: np.ndarray, 
        x_test: np.ndarray, 
        x_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @staticmethod
    def _build_preprocessing() -> Sequential:
        model = Sequential()

        model.add(Rescaling(scale=(1.0 / 255.0)))

        return model

    @staticmethod
    def _build_data_augmentation() -> Sequential:
        model = Sequential()

        model.add(RandomRotation(factor=0.08))
        model.add(RandomTranslation(height_factor=0.08, width_factor=0.08))
        model.add(RandomZoom(height_factor=0.08, width_factor=0.08))

        return model

    def _prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        shuffle: bool = False,
        augment: bool = False
    ) -> tf.data.Dataset:
        preprocessing_model = self._build_preprocessing()
        dataset = dataset.map(
            map_func=lambda x, y: (preprocessing_model(x, training=False), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1_000)

        dataset = dataset.batch(batch_size=self.batch_size)

        if augment:
            data_augmentation_model = self._build_data_augmentation()
            dataset = dataset.map(
                map_func=lambda x, y: (data_augmentation_model(x, training=False), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def _show_sample(self, n_samples: int = 1) -> None:
        batch = self.get_train_set().take(1)
        images, labels = batch.as_numpy_iterator().next()
        for idx in range(n_samples):
            x = images[idx]
            y = labels[idx]
            print(f'Class for current image: {np.argmax(y)}')
            plt.imshow(x)
            plt.show()

    def _summary(self) -> None:
        print(f'Input data shape: {self.img_shape}')
        print(f'Training Set shape: {self.train_dataset.element_spec}')
        print(f'Validation Set shape: {self.val_dataset.element_spec}')
        print(f'Test Set shape: {self.test_dataset.element_spec}')
        print(f'Amount of Samples in Training Set: {self.train_size}')
        print(f'Amount of Samples in Validation Set: {self.val_size}')
        print(f'Amount of Samples in Test Set: {self.test_size}')

if __name__ == '__main__':
    catdogsDataset = CIFAR10Dataset()
    catdogsDataset._summary()
    catdogsDataset._show_sample(n_samples=5)