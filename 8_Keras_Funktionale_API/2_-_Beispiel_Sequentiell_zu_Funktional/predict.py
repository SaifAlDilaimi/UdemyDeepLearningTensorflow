# keras, jnumpy und os
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
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

# model laden
model = load_model('catvsdog.npz')

# Teste vorhersage an zwei Beispiele
cat = load_img(os.path.join(test_data_dir, 'cat.jpg'))
cat = img_to_array(cat)  # shape (3, 150, 150)
cat /= 255.

dog = load_img(os.path.join(test_data_dir, 'dog.jpg'))
dog = img_to_array(dog)  # shape (3, 150, 150),
dog /= 255.

x = np.array([cat, dog])
prediction = model.predict(x)

for y in prediction:
    label = np.argmax(y)
    if label == 0:
        print("Bild enthält eine Katze")
    if label == 1:
        print("Bild enthält ein Hund")