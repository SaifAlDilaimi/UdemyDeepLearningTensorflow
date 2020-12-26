import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

np.random.seed(0)
tf.random.set_seed(0)

class MNISTDataset():
    def __init__(self, validation_size: float = 0.10) -> None:
        # User-definen constants
        self.num_classes = 10
        self.batch_size = 128

        # dataset laden
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Split the dataset
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size)

        # Preprocess x data
        self.x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)
        self.x_test = np.expand_dims(x_test, axis=-1).astype(np.float32)
        self.x_val = np.expand_dims(x_val, axis=-1).astype(np.float32)

        # Normalizing the RGB codes by dividing it to the max RGB value.
        self.x_train /= 255
        self.x_test /= 255
        self.x_val /= 255

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

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1_000)

        dataset = dataset.batch(batch_size=self.batch_size)

        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def _show_sample(self, n_samples: int = 1) -> None:
        batch = self.get_train_set().take(1)
        images, labels = batch.as_numpy_iterator().next()
        for idx in range(n_samples):
            x = images[idx]
            y = labels[idx]
            print(f'Class for current image: {np.argmax(y)}')
            plt.imshow(x, cmap='Greys')
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
    mnistDataset = MNISTDataset()
    mnistDataset._summary()
    mnistDataset._show_sample(n_samples=5)