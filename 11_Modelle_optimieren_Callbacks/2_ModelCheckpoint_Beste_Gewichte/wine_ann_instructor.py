# Allgemeine imports
import itertools
import os
import math
import numpy as np
import pandas as pd

# keras text verarbeitung
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Dense, Input, Flatten
from keras.layers import Concatenate, Activation

from keras.losses import mse
from keras.optimizers import Adam

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# SKLearn woerter label klasse
from sklearn.preprocessing import LabelEncoder

# statische variablen
dataset_file_name = "winemag-data_first150k.csv"
epochs = 10
learning_rate = 1e-3
batch_size = 128
vocab_size = 12000
max_seq_length = 170


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

    # validation
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

    # Shapes der sets
    print("Train size: ", x_train_description.shape, x_train_variety.shape, y_train_price.shape)
    print("Validation size: ", x_val_description.shape, x_val_variety.shape, y_val_price.shape)
    print("Test size: ", x_test_description.shape, x_test_variety.shape, y_test_price.shape)


    return (
        (x_train_description, bow_train_description, x_train_variety, x_train_embed, y_train_price), 
        (x_val_description, bow_val_description, x_val_variety, x_val_embed, y_val_price), 
        (x_test_description, bow_test_description, x_test_variety, x_test_embed, y_test_price)
    )

# Lädt das Modell und erzeugt vorhersagen für das test set
def get_predictions():
    # Modell laden
    model = load_model('wine_ann_model.h5')
    
    # Dataset laden
    (
        (x_train_description, bow_train_description, x_train_variety, x_train_embed, y_train_price), 
        (x_val_description, bow_val_description, x_val_variety, x_val_embed, y_val_price), 
        (x_test_description, bow_test_description, x_test_variety, x_test_embed, y_test_price)
    ) = prepare_dataset()

    # umwandeln zu one-hot
    num_classes = np.max(x_train_variety) + 1 # notwendig für one-hot
    x_train_variety = to_categorical(x_train_variety, num_classes=num_classes)
    x_val_variety = to_categorical(x_val_variety, num_classes=num_classes)
    x_test_variety = to_categorical(x_test_variety, num_classes=num_classes)
    
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
        actual_price = y_test_price.iloc[i]
        price_predicted = predictions[i][0]
        description = x_test_description.iloc[i]
        
        # summieren der differenz zwischen prediction und label
        diff += np.abs(price_predicted - actual_price)

        print(description)
        print("Predicted Price: ", price_predicted, "; Actual Price: ", actual_price)

    print("Allgemeine differenz zwischen prediction und laben des Testset ist: ", diff)

# Trainiert das Multi-input modell für Weinpreis vorhersage
def train():
    # subsets laden
    (
        (x_train_description, bow_train_description, x_train_variety, x_train_embed, y_train_price), 
        (x_val_description, bow_val_description, x_val_variety, x_val_embed, y_val_price), 
        (x_test_description, bow_test_description, x_test_variety, x_test_embed, y_test_price)
    ) = prepare_dataset()

    # umwandeln zu one-hot
    num_classes = np.max(x_train_variety) + 1 # notwendig für one-hot
    x_train_variety = to_categorical(x_train_variety, num_classes=num_classes)
    x_val_variety = to_categorical(x_val_variety, num_classes=num_classes)
    x_test_variety = to_categorical(x_test_variety, num_classes=num_classes)

    # Modelle definieren
    wide_model = build_wide_model(num_classes)
    deep_model = build_deep_model()

    # Beide Modelle zusammenfügen
    merged = Concatenate()([wide_model.output, deep_model.output])
    merged = Dense(1, name="price")(merged)

    inputs = wide_model.input + [deep_model.input]
    combined_model = Model(inputs=inputs, outputs=merged)
    
    # kompilieren
    opt = Adam(lr=learning_rate)
    checkpoint = ModelCheckpoint(filepath="model.{epoch:02d}-{val_loss:.2f}.h5", 
                                monitor="val_loss", 
                                mode="min", 
                                verbose=1, 
                                save_best_only=True, 
                                save_weights_only=False)
    combined_model.compile(loss=mse, optimizer=opt, metrics=['accuracy'])
    combined_model.summary()

    # trainieren
    combined_model.fit(x={
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
        }, y_val_price),
        callbacks=[checkpoint]
    )

    combined_model.save('wine_ann_model.h5')

# Erstellt ein neuronales netz das zwei Inputs annimmt
def build_wide_model(num_classes):
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
    train()
    #get_predictions()

if __name__ == '__main__':
    main()