# Allgemeine imports
import itertools
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf

# keras text verarbeitung
from keras.models import Model, load_model
from keras.layers import Embedding, Dense, Input, Flatten
from keras.layers import Concatenate, Activation

from keras.losses import mse, cosine_proximity
from keras.optimizers import Adam

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# SKLearn woerter label klasse
from sklearn.preprocessing import LabelEncoder

# import matplotlib
import matplotlib.pyplot as plt

# statische variablen



# Funktion die das Wine Dataset vorbereitet und aufteilt
def prepare_dataset():
    pass

# Lädt das Modell und erzeugt vorhersagen für das test set
def get_predictions():
    pass

# gibt die aktuelle preisdifferenz der batch
def price_difference(y_true, y_pred):
    pass
    
# Erstellt ein neuronales netz das zwei Inputs annimmt
def build_wide_model(num_classes):
    pass

# Erstellt ein tiefes neuronales netz das die beschreibungstexte verarbeitet
def build_deep_model():
    pass

# Trainiert das Multi-input modell für Weinpreis vorhersage
def train():
    pass


def main():
    if os.path.exists('wine_ann_model.npz'):
        get_predictions()
    else:
        train()

if __name__ == '__main__':
    main()