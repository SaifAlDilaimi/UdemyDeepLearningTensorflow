import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image

np.random.seed(0)
tf.random.set_seed(0)

DATA_DIR = os.path.join('C:/Users/saifa/Desktop/Dataset_CatsDogs/PetImages')
X_FILE_PATH = os.path.join(DATA_DIR, "x.npy")
Y_FILE_PATH = os.path.join(DATA_DIR, "y.npy")
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

class CATDOGDataset():
    def __init__(self, test_size: float = 0.2, validation_size: float = 0.10) -> None:
        # User-definen constants
        self.num_classes = 2
        self.batch_size = 32

        # Load the data set
        if not os.path.exists(X_FILE_PATH):
            self.extract_images()
        x = np.load(X_FILE_PATH)
        y = np.load(Y_FILE_PATH)
        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size)
        # Preprocess x data
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)
        self.x_val = x_val.astype(np.float32)
        
        # Normalizing the RGB codes by dividing it to the max RGB value.
        self.x_train /= 255.0
        self.x_test /= 255.0
        self.x_val /= 255.0

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
        self.train_dataset = self._prepare_dataset(self.train_dataset, shuffle=True, augment=True)
        self.test_dataset = self._prepare_dataset(self.test_dataset)
        self.val_dataset = self._prepare_dataset(self.val_dataset)

    def get_train_set(self) -> tf.data.Dataset:
        return self.train_dataset

    def get_test_set(self) -> tf.data.Dataset:
        return self.test_dataset

    def get_val_set(self) -> tf.data.Dataset:
        return self.val_dataset

    def extract_images(self) -> None:
        pass

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
    catdogsDataset = CATDOGDataset()
    catdogsDataset._summary()
    catdogsDataset._show_sample(n_samples=5)