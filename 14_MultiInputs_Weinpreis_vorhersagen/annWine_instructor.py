import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

from WineDatasetContainer import WineDataset

EPOCHS = 30
LEARNING_RATE = 0.0001

LOGS_DIR = os.path.abspath("C:/Users/saifa/Desktop/UdemyDeepLearningTensorflow-main/logs")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)

MODELS_DIR = os.path.abspath("C:/Users/saifa/Desktop/UdemyDeepLearningTensorflow-main/models")
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

def build_wide_model(vocab_size: int, num_variety: int) -> Model:
    # Model mit zwei Inputs: Beschreibung und varität
    bow_input = Input(shape=(vocab_size,), name="input_description")
    variety_input = Input(shape=(num_variety,), name="input_variety")

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

    model.compile(
        loss=mse, 
        optimizer=Adam(), 
        metrics=['accuracy']
    )
    model.summary()

    return model

def build_deep_model(vocab_size: int, max_seq_length: int) -> Model:
    # Model mit einem Input
    input_layer = Input(shape=(max_seq_length,), name="input_description_embed")

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

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=mse,
        metrics=['accuracy']
    )
    model.summary()

    return model

def build_model(vocab_size: int, max_seq_length: int, num_variety: int) -> Model:
    # Modelle definieren
    wide_model = build_wide_model(vocab_size, num_variety)
    deep_model = build_deep_model(vocab_size, max_seq_length)

    # Beide Modelle zusammenfügen
    merged = Concatenate()([wide_model.output, deep_model.output])
    merged = Dense(1, name="price")(merged)

    inputs = wide_model.input + [deep_model.input]
    combined_model = Model(inputs=inputs, outputs=merged)

    return combined_model

def train(model_name: str, data: WineDataset) -> None:
    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()

    model = build_model(data.vocab_size, data.max_seq_length, data.num_variety)

    model.compile(
        loss=mse,
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy", price_difference]
    )

    model.summary()

    model_log_dir = os.path.join(LOGS_DIR, f"model_{model_name}")

    tb_callback = TensorBoard(
        log_dir=model_log_dir,
        histogram_freq=0,
        profile_batch=0,
        write_graph=True
    )

    mc_callback = ModelCheckpoint(
        os.path.join(MODELS_DIR, model_name),
        monitor='val_loss',
        mode='auto',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_accuracy', # auch val_accuracy probieren
        mode='auto',
        factor=0.9,
        patience=3,
        cooldown=0,
        min_lr=0,
        verbose=1
    )

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        batch_size=data.batch_size,
        verbose=1,
        validation_data=val_dataset,
        callbacks=[tb_callback, mc_callback, reduce_lr_callback]
    )

def evaluate(model_name: str, data: WineDataset) -> None:
    model_path = os.path.join(MODELS_DIR, model_name)

    model = load_model(model_path, custom_objects={'price_difference': price_difference})

    test_dataset = data.get_test_set()
    batch = test_dataset.take(1)
    _, y = batch.as_numpy_iterator().next()

    y_pred = model.predict(test_dataset)

    # Vorhersage testen
    max_number_prediction = 40
    diff = 0.0

    sample_prices = y[:max_number_prediction]
    sample_pred = y_pred[:max_number_prediction]
    sample_descr = data.x_test_description[:max_number_prediction]

    for y, y_pred, description in zip(sample_prices, sample_pred, sample_descr):
        # summieren der differenz zwischen prediction und label
        diff += np.abs(y_pred[0] - y)

        print('-----------------------------------------------------')
        print(description)
        print("Predicted Price: ", y_pred[0], "; Actual Price: ", y)

    score = model.evaluate(test_dataset)
    print(f'The absolute price difference for the first {max_number_prediction} samples is {diff:.2f}$ and the loss is {score[0]:.3f}')

# gibt die aktuelle preisdifferenz der batch
def price_difference(y_true, y_pred):
    x = tf.expand_dims(y_true, -1)
    y = tf.expand_dims(y_pred, -1)
    diff = tf.subtract(x, y)
    diff = tf.abs(diff)
    diff = tf.reduce_sum(diff)
    return diff

if __name__ == "__main__":
    data = WineDataset()

    model_name = f"wine_wide_deep"

    if not os.path.exists(os.path.join(MODELS_DIR, model_name)):
        train(model_name, data)

    evaluate(model_name, data)
