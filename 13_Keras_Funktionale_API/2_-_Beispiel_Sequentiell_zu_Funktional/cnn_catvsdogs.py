# keras, jnumpy und os
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization, Flatten, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import RMSprop

# statische variablen
img_width, img_height = 150, 150
working_dir = os.path.join(os.path.curdir, "images")
train_data_dir = os.path.join(working_dir, "train")
validate_data_dir = os.path.join(working_dir, "validation")
test_data_dir = os.path.join(working_dir, "test")
nb_train_samples = 5000
nb_validation_samples = 1000

# Hyperparameter
epochs = 50
batch_size = 16
learning_rate = 0.0001
input_shape = (img_width, img_height, 3)

# model definieren
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(2))
model.add(Activation('softmax'))

# model kompilieren
opt = RMSprop(lr=learning_rate)
model.compile(loss=binary_crossentropy, optimizer=opt, metrics=['accuracy'])
model.summary()

# bilder transformieren
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

path = os.path.join(os.path.join(validate_data_dir, 'cat'), 'cat.2501.jpg')
img = load_img(path)
x = img_to_array(img) # shape (150, 150, 3)
x = x.reshape((1,)+x.shape) # (1, 150, 150, 3)

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir=os.path.join(working_dir, 'preview'), save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break

# training und validation generator
train_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validate_data_generator = ImageDataGenerator(rescale=1./255)

# generatror iterieren
train_gen = train_data_generator.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    class_mode='categorical'
)

validation_gen = validate_data_generator.flow_from_directory(
    validate_data_dir,
    target_size=(img_height, img_width),
    class_mode='categorical'
)

labels = train_gen.class_indices
print(labels)

# trainieren
history = model.fit_generator(
    train_gen,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_gen,
    validation_steps=nb_validation_samples // batch_size
)

# model speichern
model.save("catvsdog.npz")

score = model.evaluate_generator(validation_gen)
print("Accuracy on val: ", score[1])