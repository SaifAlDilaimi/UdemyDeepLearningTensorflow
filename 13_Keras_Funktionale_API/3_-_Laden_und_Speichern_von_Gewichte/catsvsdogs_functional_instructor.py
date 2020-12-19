# keras imports vorbereiten
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization, Flatten, Dense, Input
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.optimizers import RMSprop

# Statische variablen setzen
img_width, img_height = 150, 150
working_dir = "E:/Keras_Kurs/working_dir/7/catsvsdogs/data/"
train_data_dir = "E:/Keras_Kurs/working_dir/7/catsvsdogs/data/train/"
validate_data_dir = "E:/Keras_Kurs/working_dir/7/catsvsdogs/data/validation/"
test_data_dir = "E:/Keras_Kurs/working_dir/7/catsvsdogs/data/test/"
nb_train_samples = 5000
nb_validation_samples = 1000

# Hyperparameter
epochs = 20
batch_size = 16
learning_rate = 1e-3 # 0.0001
input_shape = (img_width, img_height, 3)

def predict_catvsdogs(x):
    model = load_model("catvsdogs.h5")
    prediction = model.predict(x)
    prediction = np.floor(prediction * 100) / 100
    print("Die Bilder enthalten: ", prediction)

def train():        
    # model definieren
    input_layer = Input(shape=input_shape, name="input_image")

    x = Conv2D(32, (3,3), name="conv1_32")(input_layer)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3,3), name="conv2_32")(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3,3), name="conv1_64")(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(2)(x)

    predictions = Activation('softmax')(x)
    model = Model(inputs=input_layer, outputs=predictions)
    model.load_weights(filepath='catvsdogs.h5')

    # Model kompilieren
    opt = RMSprop(lr=learning_rate)
    model.compile(loss=binary_crossentropy, optimizer=opt, metrics=['accuracy'])
    model.summary()

    # Bilder transformieren und augmentieren zum testen
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    img = load_img(validate_data_dir + 'cats/cat.2501.jpg')  # lädt es in ein PIL image
    x = img_to_array(img)  # shape (150, 150, 3)
    x = x.reshape((1,) + x.shape)  # shape (1, 150, 150, 3)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=working_dir + 'preview', save_prefix='cat', save_format='jpeg'):
        i += 1
        if i > 20:
            break  # sonst läuft endlos

    # training und validation augmentierer erstellen
    train_data_generator = ImageDataGenerator(rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    validation_data_generator = ImageDataGenerator(rescale=1. /255)

    # Generatoren erstellen für das training process
    train_gen = train_data_generator.flow_from_directory(train_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

    validation_gen = validation_data_generator.flow_from_directory(validate_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

    # labels unser Klassifizierungsproblem
    labels = train_gen.class_indices
    print(labels)
    
    # Model trainieren
    history = model.fit_generator(train_gen,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_gen,
                    validation_steps=nb_validation_samples // batch_size)

    score = model.evaluate_generator(validation_gen)
    print("score Acc: ", score[1])
    
    # Model speichern
    model.save("catvsdogs.h5")

def main():
    train()

    # teste vorhersage an zweiten Beispiel
    cat = load_img(test_data_dir + 'cat.jpg')  # lädt es in ein PIL image
    cat = img_to_array(cat)  # shape (3, 150, 150)
    cat /= 255.
    
    cat2 = load_img(test_data_dir + 'cat2.jpg')  # lädt es in ein PIL image
    cat2 = img_to_array(cat2)  # shape (3, 150, 150)
    cat2 /= 255.

    cat3 = load_img(test_data_dir + 'cat3.jpg')  # lädt es in ein PIL image
    cat3 = img_to_array(cat3)  # shape (3, 150, 150)
    cat3 /= 255.

    dog = load_img(test_data_dir + 'dog.jpg')  # lädt es in ein PIL image
    dog = img_to_array(dog)  # shape (3, 150, 150),
    dog /= 255.
    
    dog2 = load_img(test_data_dir + 'dog2.jpg')  # lädt es in ein PIL image
    dog2 = img_to_array(dog2)  # shape (3, 150, 150),
    dog2 /= 255.

    dog3 = load_img(test_data_dir + 'dog3.jpg')  # lädt es in ein PIL image
    dog3 = img_to_array(dog3)  # shape (3, 150, 150),
    dog3 /= 255.

    x = np.array([cat, cat2, cat3, dog, dog2, dog3])
    predict_catvsdogs(x)

if __name__ == '__main__':
    main()