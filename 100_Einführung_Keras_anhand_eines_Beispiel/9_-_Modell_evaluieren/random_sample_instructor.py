# importieren von keras und numpy
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical

# dataset vorbereiten
# generieren von train, val und test daten
x_train = np.random.random((1000, 100))
x_val = np.random.random((100, 100))
x_test = np.random.random((100, 100))

# generieren von train, val und test labels
y_train = np.random.randint(0, 10, size=(1000, 1))
y_val = np.random.randint(0, 10, size=(100, 1))
y_test = np.random.randint(0, 10, size=(100, 1))

# Test ausgabe
print(f"Shape von x_train: {x_train.shape}")
print(f"Shape von y_train: {y_train.shape}")

# model definieren
model = Sequential()

# model layers definieren
model.add(Dense(32, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

# model kompilieren
rmsprop_opt = RMSprop(lr=0.0001)
model.compile(optimizer=rmsprop_opt, 
            loss=categorical_crossentropy, 
            metrics=['accuracy'])
model.summary()

# model labels zu one-hot definieren
y_train_one_hot = to_categorical(y_train, num_classes=10)
y_val_one_hot = to_categorical(y_val, num_classes=10)
y_test_one_hot = to_categorical(y_test, num_classes=10)

# Test ausgabe
print("Shape von x_train: ", x_train.shape)
print("Shape von y_train: ", y_train_one_hot.shape)

# model trainieren
history = model.fit(x_train, y_train_one_hot, 
        batch_size=32, 
        epochs=10, 
        validation_data=(x_val, y_val_one_hot))


# training ergebnisse evaluieren, overfitting / underfitting?
loss_training = history.history['loss']
acc_training = history.history['accuracy']
loss_val = history.history['val_loss']
acc_val = history.history['val_accuracy']

# plotte ergebnisse
epochs = range(10)
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

# model evaluieren
score = model.evaluate(x_test, y_test_one_hot, batch_size=32)
print(score)

# model nutzen, um etwas vorherzusagen
x = np.random.random((1, 100))
y = model.predict(x, batch_size=32) # gibt die wahrscheinlichkeiten jeder Klasse aus
print(f"Prediction of x: {y}")
class_idx = np.argmax(y)
print(f'Class with highest prob. {class_idx}')