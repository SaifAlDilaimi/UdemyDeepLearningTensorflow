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

    def extract_images(self) -> None:
        cats_dir = os.path.join(DATA_DIR, "Cat")
        dogs_dir = os.path.join(DATA_DIR, "Dog")

        dirs = [cats_dir, dogs_dir]
        class_names = ["cat", "dog"]

        for d in dirs:
            for f in os.listdir(d):
                if f.split(".")[-1] != "jpg":
                    print(f"Removing file: {f}")
                    os.remove(os.path.join(d, f))

        num_cats = len(os.listdir(cats_dir))
        num_dogs = len(os.listdir(dogs_dir))
        num_images = num_cats + num_dogs

        x = np.zeros(
            shape=(num_images, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            dtype=np.float32
        )
        y = np.zeros(
            shape=(num_images,),
            dtype=np.float32
        )

        cnt = 0
        for d, class_name in zip(dirs, class_names):
            print(f'Reading images of {class_name}...')
            for f in os.listdir(d):
                img_file_path = os.path.join(d, f)
                try:
                    im = Image.open(img_file_path)
                    im = im.resize((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS)
                    x[cnt] = np.array(im, dtype=np.float32).reshape(IMG_SHAPE)
                    if class_name == "cat":
                        y[cnt] = 0
                    elif class_name == "dog":
                        y[cnt] = 1
                    else:
                        print("Invalid class name!")
                    cnt += 1
                except:  # noqa: E722
                    print(f"Image {f} can't be read!")
                    os.remove(img_file_path)

        # Dropping not readable image idxs
        x = x[:cnt]
        y = y[:cnt]

        np.save(X_FILE_PATH, x)
        np.save(Y_FILE_PATH, y)

    def data_augmentation(self, augment_size: int = 5_000) -> None:
        image_generator = ImageDataGenerator(
            rotation_range=25,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
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
            plt.imshow(x_aug)
            plt.subplot(122)
            plt.title('Original')
            plt.imshow(x)
            plt.show()
        # Append the augmented images to the train set
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]

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