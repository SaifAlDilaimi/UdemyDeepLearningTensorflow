# confusion matrix
from confusion_matrix import plot_confusion_matrix

# import sklearn, numpy und keras
from sklearn.metrics import confusion_matrix

import numpy as np
import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

# import matplotlib
import matplotlib.pyplot as plt



# Statische variablen festlegen
epochs = 5
batch_size = 128
learning_rate = 1e-3
momentum = 0.9

# input dimensions
img_rows, img_cols = 28, 28

# mnist dataset laden
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# dataset umstrukturieren
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# normieren auf [0, 1] on [0, 255]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Info ausgaben zum dataset
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# model labels zu one-hot definieren
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=10)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=10)


# Model+layers definieren
model = Sequential()
# Input: Bilder in der größe 28x28 mit 1 channel -> (28, 28, 1)
model.add(Conv2D(32, (3, 3), input_shape=(img_rows, img_cols, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

# kompilieren und trainieren
sgd_opt = SGD(lr=learning_rate, momentum=momentum, nesterov=True)
model.compile(loss=categorical_crossentropy, optimizer=sgd_opt, metrics=['accuracy'])

history = model.fit(x_train, y_train_one_hot, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test_one_hot), verbose=1)

# evaluieren
score = model.evaluate(x_test, y_test_one_hot, batch_size=batch_size)
print(score)
# training ergebnisse
loss_training = history.history['loss']
acc_training = history.history['acc']
loss_val = history.history['val_loss']
acc_val = history.history['val_acc']

# plotte ergebnisse
epochs = range(epochs)
plt.plot(epochs, loss_training, label="training loss")
plt.plot(epochs, loss_val, label="validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.plot(epochs, acc_training, label="training accuracy")
plt.plot(epochs, acc_val, label="validation accuracy")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()