import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(0)
tf.random.set_seed(0)

DATA_DIR = os.path.join('C:/Users/saifa/Desktop/Dataset_Weinpreise')
CSV_PATH = os.path.join(DATA_DIR, 'Dataset_Weinpreise.csv')
X_FILE_PATH = os.path.join(DATA_DIR, "x.npz")
Y_FILE_PATH = os.path.join(DATA_DIR, "y.npy")

class WineDataset():
    def __init__(self, test_size: float = 0.2, validation_size: float = 0.10) -> None:
        # User-definen constants
        self.num_variety = 40
        self.batch_size = 32
        self.vocab_size = 12000
        self.max_seq_length = 170
        self.variety_relevance_limit = 500

        # Load the data set
        if not os.path.exists(X_FILE_PATH):
            self.extract_csv()

        with np.load(X_FILE_PATH, allow_pickle=True) as data:
            x_description = data['description']
            x_variety = data['variety']
        y = np.load(Y_FILE_PATH)

        self.train_size = int(len(x_description) * (1 - test_size - validation_size))
        self.test_size = int(len(x_description) * test_size)
        self.val_size = int(len(x_description) * validation_size)

        train_limit = self.train_size
        val_limit = self.train_size + self.val_size

        # Split the dataset
        self.x_train_description = x_description[:train_limit]
        self.x_train_variety = x_variety[:train_limit]
        self.y_train = y[:train_limit]

        # validation
        self.x_val_description = x_description[train_limit:val_limit]
        self.x_val_variety = x_variety[train_limit:val_limit]
        self.y_val = y[train_limit:val_limit]

        # test
        self.x_test_description = x_description[val_limit:]
        self.x_test_variety = x_variety[val_limit:]
        self.y_test = y[val_limit:]

        # Tokenizer erstellen, sodass die wörter in der beschreibung trainiert werden
        tokenize = Tokenizer(num_words=self.vocab_size, char_level=False)
        tokenize.fit_on_texts(self.x_train_description) # wir trainieren nur an train set, weil es das größte ist

        # (feature1) erzeugen von bag of words (bow) der beschreibung für train, val und test
        self.x_train_bow_description = tokenize.texts_to_matrix(self.x_train_description)
        self.x_val_bow_description = tokenize.texts_to_matrix(self.x_val_description)
        self.x_test_bow_description = tokenize.texts_to_matrix(self.x_test_description)

        # (feature2) Erzeugn von one-hot vektoren für die varietät mithilfe von SKLearn
        encoder = LabelEncoder()
        encoder.fit(self.x_train_variety)
        self.x_train_variety = encoder.transform(self.x_train_variety)
        self.x_val_variety = encoder.transform(self.x_val_variety)
        self.x_test_variety = encoder.transform(self.x_test_variety)

        self.x_train_variety = to_categorical(self.x_train_variety, num_classes=self.num_variety)
        self.x_test_variety = to_categorical(self.x_test_variety, num_classes=self.num_variety)
        self.x_val_variety = to_categorical(self.x_val_variety, num_classes=self.num_variety)

        # (feature 3) erzeugen von embeddings der beschreibungen
        x_train_embed = tokenize.texts_to_sequences(self.x_train_description)  
        x_val_embed = tokenize.texts_to_sequences(self.x_val_description)  
        x_test_embed = tokenize.texts_to_sequences(self.x_test_description)

        # Normalisieren der sequenzen
        self.x_train_embed = pad_sequences(x_train_embed, maxlen=self.max_seq_length, padding="post")
        self.x_val_embed = pad_sequences(x_val_embed, maxlen=self.max_seq_length, padding="post")
        self.x_test_embed = pad_sequences(x_test_embed, maxlen=self.max_seq_length, padding="post")

        # tf.data Datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                'input_description': self.x_train_bow_description, 
                'input_description_embed': self.x_train_embed, 
                'input_variety': self.x_train_variety
                },
                self.y_train
            )
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    'input_description': self.x_test_bow_description, 
                    'input_description_embed': self.x_test_embed, 
                    'input_variety': self.x_test_variety
                },
                self.y_test
            )
        )
        self.val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    'input_description': self.x_val_bow_description, 
                    'input_description_embed': self.x_val_embed, 
                    'input_variety': self.x_val_variety
                },
                self.y_val
            )
        )
        self.train_dataset = self._prepare_dataset(self.train_dataset, shuffle=True)
        self.test_dataset = self._prepare_dataset(self.test_dataset)
        self.val_dataset = self._prepare_dataset(self.val_dataset)

    def get_train_set(self) -> tf.data.Dataset:
        return self.train_dataset

    def get_test_set(self) -> tf.data.Dataset:
        return self.test_dataset

    def get_val_set(self) -> tf.data.Dataset:
        return self.val_dataset

    def extract_csv(self) -> None:
        # CSV inhalt zu Panda Data Frame umwandeln
        data = pd.read_csv(CSV_PATH)
        # data shuffeln/mischen
        data = data.sample(frac=1)

        # Leere felder filtern und zeilen limitieren
        data = data[pd.notnull(data['country'])]
        data = data[pd.notnull(data['price'])]
        data = data.drop(data.columns[0], axis=1)

        # variatät nach relevanz filtern
        value_counts = data['variety'].value_counts()
        remove_rows = value_counts[value_counts <= self.variety_relevance_limit].index
        data.replace(remove_rows, np.nan, inplace=True)
        data = data[pd.notnull(data['variety'])]

        x_description = data['description'].values
        x_variety = data['variety'].values
        y = data['price'].values

        np.savez(X_FILE_PATH, description=x_description, variety=x_variety)
        np.save(Y_FILE_PATH, y)

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
        x, price = batch.as_numpy_iterator().next()

        for idx in range(n_samples):
            x_description = self.x_train_description[idx]
            x_variety = x['input_variety'][idx]
            y = price[idx]
            print('------------------------------------------')
            print(f'Description: {x_description}')
            print(f'Variety: {np.argmax(x_variety)}')
            print(f'Price: {y}')
            print('------------------------------------------')

    def _summary(self) -> None:
        print(f'Input data shape (Feature 1): {self.x_train_bow_description.shape}')
        print(f'Input data shape (Feature 2): {self.x_train_embed.shape}')
        print(f'Input data shape (Feature 3): {self.x_train_variety.shape}')
        print(f'Training Set shape: {self.train_dataset.element_spec}')
        print(f'Validation Set shape: {self.val_dataset.element_spec}')
        print(f'Test Set shape: {self.test_dataset.element_spec}')
        print(f'Amount of Samples in Training Set: {self.train_size}')
        print(f'Amount of Samples in Validation Set: {self.val_size}')
        print(f'Amount of Samples in Test Set: {self.test_size}')

if __name__ == '__main__':
    wineData = WineDataset()
    wineData._summary()
    wineData._show_sample(n_samples=5)