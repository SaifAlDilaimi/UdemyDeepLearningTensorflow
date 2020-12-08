# confusion matrix
from confusion_matrix import plot_confusion_matrix

# import sklearn, numpy, keras
from sklearn.metrics import confusion_matrix

import numpy as np
import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from LRTensorBoard import LRTensorBoard
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

# import matplotlib
import matplotlib.pyplot as plt

# Statische variablen
epochs = 5
batch_size = 128
learning_rate = 1e-3
momentum = 0.9

# input dimension 
img_rows, img_cols  = 28, 28

# mnist laden
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# daten umstrukturieren
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# nomieren unsere daten
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# [0, 255] -> [0,1]
x_train /= 255
x_test /= 255

# infos ausgeben
print("Shape von x_train: ", x_train.shape)
print(x_train.shape[0], "train sample")
print(x_test.shape[0], "test sample")

# labels -> 1 hot vektoren
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=10)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=10)

# model + layers
model = Sequential()
# input: Bilder in der grtöße 28x29 mit 1 channel
model.add(Conv2D(32, (3,3), input_shape=(img_rows, img_cols, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

# model trainieren
sgd_opt = SGD(lr=learning_rate, momentum=momentum, nesterov=True)
model.compile(loss=categorical_crossentropy, optimizer=sgd_opt, metrics=['accuracy'])


embedding_layer_names = set(layer.name
                            for layer in model.layers
                            if layer.name.startswith('dense_'))

with open("./logs/metadata.tsv", "w") as f:
    np.savetxt(f, y_test[:2000])

tb = LRTensorBoard(log_dir="./logs",
                histogram_freq=1, batch_size=32,
                write_graph=True, write_images=True,
                embeddings_freq=1, embeddings_metadata="metadata.tsv",
                embeddings_layer_names=embedding_layer_names,
                embeddings_data=x_test[:2000])

history = model.fit(x_train, y_train_one_hot, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test_one_hot), callbacks=[tb])

# evaluieren
score = model.evaluate(x_test, y_test_one_hot, batch_size=batch_size)
print(score)

# ergebnisse speichern
loss_training = history.history['loss']
acc_training = history.history['acc']
loss_val = history.history['val_loss']
acc_val = history.history['val_acc']

# Plotting
epochs = range(epochs)
plt.plot(epochs, loss_training, label="training loss")
plt.plot(epochs, loss_val, label="val loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(epochs, acc_training, label="training acc")
plt.plot(epochs, acc_val, label="val acc")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.show()

# plotting von confusion matrix
pre_cls = model.predict_classes(x_test)
cnf_matrix = confusion_matrix(y_test, pre_cls)
class_names = [str(cl) for cl in range(10)]
print(cnf_matrix)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title="CNF Matrix MNIST")
plt.show()