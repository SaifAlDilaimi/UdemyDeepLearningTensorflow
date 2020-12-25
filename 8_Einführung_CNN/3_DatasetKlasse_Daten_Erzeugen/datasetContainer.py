import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

np.random.seed(0)
tf.random.set_seed(0)

class RandomImageDataset():
    def __init__(self, validation_size: float = 0.22) -> None:
        pass

    def get_train_set(self) -> tf.data.Dataset:
        pass

    def get_test_set(self) -> tf.data.Dataset:
        pass

    def get_val_set(self) -> tf.data.Dataset:
        pass

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