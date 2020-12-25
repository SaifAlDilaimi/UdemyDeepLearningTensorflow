import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

np.random.seed(0)
tf.random.set_seed(0)

class RandomImageDataset():
    def __init__(self, validation_size: float = 0.22) -> None:
        # User-definen constants
        self.num_classes = 10
        self.batch_size = 128

        # dataset generieren -> Bilder 1000x100x100x3
        # generieren von train und test daten
        x_train = np.random.random((1000, 100, 100, 3))
        x_test = np.random.random((200, 100, 100, 3))

        # generieren von train und test labels
        y_train = np.random.randint(10, size=(1000, 1))
        y_test = np.random.randint(10, size=(200, 1))
        # Split the dataset
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

        # tf.data Datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        self.train_dataset = self._prepare_dataset(self.train_dataset, shuffle=True)
        self.test_dataset = self._prepare_dataset(self.test_dataset)
        self.val_dataset = self._prepare_dataset(self.val_dataset)

    def get_train_set(self) -> tf.data.Dataset:
        return self.train_dataset

    def get_test_set(self) -> tf.data.Dataset:
        return self.test_dataset

    def get_val_set(self) -> tf.data.Dataset:
        return self.val_dataset

    def _prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        shuffle: bool = False,
        augment: bool = False
    ) -> tf.data.Dataset:
        pass

    def _show_sample(self, n_samples: int = 1) -> None:
        pass

    def _summary(self) -> None:
        pass

if __name__ == '__main__':
    randomImageData = RandomImageDataset()
    randomImageData._summary()
    randomImageData._show_sample(n_samples=5)