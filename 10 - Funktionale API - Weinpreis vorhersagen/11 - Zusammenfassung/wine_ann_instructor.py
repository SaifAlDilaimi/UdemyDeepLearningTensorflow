# Allgemeine imports
import itertools
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf

# keras text verarbeitung
from keras.models import Model, load_model
from keras.layers import Embedding, Dense, Input, Flatten, LSTM
from keras.layers import Concatenate, Activation

from keras.losses import mse
from keras.optimizers import Adam

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# SKLearn woerter label klasse
from sklearn.preprocessing import LabelEncoder

# import matplotlib
import matplotlib.pyplot as plt

# statische variablen
dataset_file_name = "winemag-data_first150k.csv"
model_name = "wine_ann_model.h5"
epochs = 10
learning_rate = 1e-3
batch_size = 128
vocab_size = 12000
max_seq_length = 170
num_classes = 40

# gibt die aktuelle preisdifferenz der batch
def price_difference(y_true, y_pred):
    x = tf.expand_dims(y_true, -1)
    y = tf.expand_dims(y_pred, -1)
    diff = tf.subtract(x, y)
    diff = tf.abs(diff)
    diff = tf.reduce_sum(diff)
    return diff

# Funktion die das Wine Dataset vorbereitet und aufteilt
def prepare_dataset():
    # CSV inhalt zu Panda Data Frame umwandeln
    data = pd.read_csv(dataset_file_name)
    # data shuffeln/mischen
    data = data.sample(frac=1)

    # test ausgabe
    print(data.head())

    # Leere felder filtern und zeilen limitieren
    data = data[pd.notnull(data['country'])]
    data = data[pd.notnull(data['price'])]
    data = data.drop(data.columns[0], axis=1)

    # test ausgabe
    print(data.head())

    # variatät nach relevanz filtern
    variety_limit = 500
    value_counts = data['variety'].value_counts()
    remove_rows = value_counts[value_counts <= variety_limit].index
    data.replace(remove_rows, np.nan, inplace=True)
    data = data[pd.notnull(data['variety'])]

    # test ausgabe
    print(data.head())

    # Aufteilen in train, validation und test sets
    train_size = int(len(data) * 0.8)
    validation_size = int(len(data) * 0.1)
    test_size = int(len(data) * 0.1)

    # features und labels vorbereiten
    x_train_description = data['description'][:train_size]
    x_train_variety = data['variety'][:train_size]
    y_train_price = data['price'][:train_size]

    # validation
    x_val_description = data['description'][train_size:train_size+validation_size]
    x_val_variety = data['variety'][train_size:train_size+validation_size]
    y_val_price = data['price'][train_size:train_size+validation_size]

    # test
    x_test_description = data['description'][train_size+validation_size:]
    x_test_variety = data['variety'][train_size+validation_size:]
    y_test_price = data['price'][train_size+validation_size:]

    # Tokenizer erstellen, sodass die wörter in der beschreibung trainiert werden
    tokenize = Tokenizer(num_words=vocab_size, char_level=False)
    tokenize.fit_on_texts(x_train_description) # wir trainieren nur an train set, weil es das größte ist

    # (feature1) erzeugen von bag of words (bow) der beschreibung für train, val und test
    bow_train_description = tokenize.texts_to_matrix(x_train_description)
    bow_val_description = tokenize.texts_to_matrix(x_val_description)
    bow_test_description = tokenize.texts_to_matrix(x_test_description)

    # (feature2) Erzeugn von one-hot vektoren für die varietät mithilfe von SKLearn
    encoder = LabelEncoder()
    encoder.fit(x_train_variety)
    x_train_variety = encoder.transform(x_train_variety)
    x_val_variety = encoder.transform(x_val_variety)
    x_test_variety = encoder.transform(x_test_variety)

    # (feature 3) erzeugen von embeddings der beschreibungen
    x_train_embed = tokenize.texts_to_sequences(x_train_description)  
    x_val_embed = tokenize.texts_to_sequences(x_val_description)  
    x_test_embed = tokenize.texts_to_sequences(x_test_description)

    # Normalisieren der sequenzen
    x_train_embed = pad_sequences(x_train_embed, maxlen=max_seq_length, padding="post")  
    x_val_embed = pad_sequences(x_val_embed, maxlen=max_seq_length, padding="post")  
    x_test_embed = pad_sequences(x_test_embed, maxlen=max_seq_length, padding="post")  

    x_train_variety = to_categorical(x_train_variety, num_classes=num_classes)
    x_val_variety = to_categorical(x_val_variety, num_classes=num_classes)
    x_test_variety = to_categorical(x_test_variety, num_classes=num_classes)

    # Shapes der sets
    print("Train size: ", x_train_description.shape, x_train_variety.shape, y_train_price.shape)
    print("Validation size: ", x_val_description.shape, x_val_variety.shape, y_val_price.shape)
    print("Test size: ", x_test_description.shape, x_test_variety.shape, y_test_price.shape)

    np.save('x_train_description', x_train_description)
    np.save('bow_train_description', bow_train_description)
    np.save('x_train_variety', x_train_variety)
    np.save('x_train_embed', x_train_embed)
    np.save('y_train_price', y_train_price)

    np.save('x_val_description', x_val_description)
    np.save('bow_val_description', bow_val_description)
    np.save('x_val_variety', x_val_variety)
    np.save('x_val_embed', x_val_embed)
    np.save('y_val_price', y_val_price)

    np.save('x_test_description', x_test_description)
    np.save('bow_test_description', bow_test_description)
    np.save('x_test_variety', x_test_variety)
    np.save('x_test_embed', x_test_embed)
    np.save('y_test_price', y_test_price)

# Lädt das Modell und erzeugt vorhersagen für das test set
def get_predictions():
    # Modell laden
    model = load_model('wine_ann_model.h5', custom_objects={'price_difference': price_difference})
    
    # Dataset laden
    x_test_description = np.load('x_test_description.npy')
    bow_test_description = np.load('bow_test_description.npy')
    x_test_variety = np.load('x_test_variety.npy')
    x_test_embed = np.load('x_test_embed.npy')
    y_test_price = np.load('y_test_price.npy')
    
    # Vorhersage testen
    max_number_prediction = 40
    diff = 0

    predictions = model.predict({
        'input_description': bow_test_description,
        'input_variety': x_test_variety,
        'input_description_pad': x_test_embed
    })

    print(predictions)

    for i in range(max_number_prediction):
        actual_price = y_test_price[i]
        price_predicted = predictions[i][0]
        description = x_test_description[i]
        
        # summieren der differenz zwischen prediction und label
        diff += np.abs(price_predicted - actual_price)

        print(description)
        print("Predicted Price: ", price_predicted, "; Actual Price: ", actual_price)

    print("Allgemeine differenz zwischen prediction und label des Testset ist: ", diff)

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	plt.subplot(311)
	plt.title('Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')

	# plot accuracy
	plt.subplot(312)
	plt.title('Classification Accuracy')
	plt.plot(history.history['acc'], color='blue', label='train')
	plt.plot(history.history['val_acc'], color='orange', label='test')

	# plot price diff
	plt.subplot(313)
	plt.title('Price difference')
	plt.plot(history.history['price_difference'], color='blue', label='train')
	plt.plot(history.history['val_price_difference'], color='orange', label='test')

	# save plot to file
	plt.savefig('history_plot.png', dpi=180)
	plt.close()

# Trainiert das Multi-input modell für Weinpreis vorhersage
def train():    
    # Dataset laden
    bow_train_description = np.load('bow_train_description.npy')
    x_train_variety = np.load('x_train_variety.npy') 
    x_train_embed = np.load('x_train_embed.npy')
    y_train_price = np.load('y_train_price.npy')
    
    bow_val_description = np.load('bow_val_description.npy')
    x_val_variety = np.load('x_val_variety.npy')
    x_val_embed = np.load('x_val_embed.npy')
    y_val_price = np.load('y_val_price.npy')

    # Modelle definieren
    wide_model = build_wide_model()
    deep_model = build_deep_model()

    # Beide Modelle zusammenfügen
    merged = Concatenate()([wide_model.output, deep_model.output])
    merged = Dense(1, name="price")(merged)

    inputs = wide_model.input + [deep_model.input]
    combined_model = Model(inputs=inputs, outputs=merged)
    
    # kompilieren
    opt = Adam(lr=learning_rate)
    combined_model.compile(loss=mse, optimizer=opt, metrics=['accuracy', price_difference])
    combined_model.summary()

    # trainieren
    history = combined_model.fit(x={
        'input_description': bow_train_description,
        'input_variety': x_train_variety,
        'input_description_pad': x_train_embed
    }, y={
        'price': y_train_price
    },  epochs=epochs, 
        batch_size=batch_size,
        verbose=1,
        validation_data=({
            'input_description': bow_val_description,
            'input_variety': x_val_variety,
            'input_description_pad': x_val_embed
        }, y_val_price)
    )

    combined_model.save(model_name)
    summarize_diagnostics(history)

# Erstellt ein neuronales netz das zwei Inputs annimmt
def build_wide_model():
    # Model mit zwei Inputs: Beschreibung und varität
    bow_input = Input(shape=(vocab_size,), name="input_description")
    variety_input = Input(shape=(num_classes,), name="input_variety")

    # netz layers
    x = Concatenate()([bow_input, variety_input])
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    predictions = Dense(1)(x)

    model = Model(inputs=[bow_input, variety_input], outputs=predictions)

    # kompilieren
    opt = Adam(lr=learning_rate)
    model.compile(loss=mse, optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model

# Erstellt ein tiefes neuronales netz das die beschreibungstexte verarbeitet
def build_deep_model():
    # Model mit einem Input
    input_layer = Input(shape=(max_seq_length,), name="input_description_pad")

    # netz layers
    x = Embedding(vocab_size, 8, input_length=max_seq_length)(input_layer)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation('relu')(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    out = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=out)

    # kompilieren
    opt = Adam(lr=learning_rate)
    model.compile(loss=mse, optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model

def main():
    if not os.path.exists('x_train_description.npy'):
        prepare_dataset()
    if not os.path.exists(model_name):
        train()
    
    get_predictions()

if __name__ == '__main__':
    main()