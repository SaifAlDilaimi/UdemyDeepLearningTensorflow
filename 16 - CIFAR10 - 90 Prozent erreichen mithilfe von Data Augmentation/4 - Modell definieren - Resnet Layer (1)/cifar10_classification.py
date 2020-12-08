from keras.datasets import cifar10

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Add

from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2

from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

from matplotlib import pyplot as plt

import numpy as np
import os

learning_rate = 1e-3
min_learning_rate = 0.5e-6
batch_size = 128
epochs = 200
n = 3
depth = n * 6 + 2
num_classes = 10
model_name = 'keras_cifar10_trained_model.h5'

def show_samples(X):
    # Show 9 images
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        x_i = X[i].reshape(32, 32, 3)
        plt.imshow(x_i)
    # show the plot
    plt.show()
    
def lr_schedule(epoch):
    pass

def prepare_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    show_samples(x_train)
    
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    #z-score
    mean = np.mean(x_train, axis=(0,1,2,3))
    std = np.std(x_train, axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    #data augmentation
    augment_size = 15000
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train, augment=True)

    # get transformed images
    randidx = np.random.randint(x_train.shape[0], size=augment_size)
    x_augmented = x_train[randidx].copy()
    y_augmented = y_train[randidx].copy()
    x_augmented = datagen.flow(x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]
    # append augmented data to trainset
    x_train = np.concatenate((x_train, x_augmented))
    y_train = np.concatenate((y_train, y_augmented)) 
        
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    np.save("x_train", x_train)
    np.save("y_train", y_train)
    np.save("x_test", x_test)
    np.save("y_test", y_test)

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def create_model(input_shape):
    pass

def train():
    pass

# plot diagnostic learning curves
def summarize_diagnostics(history):
    pass

def evaluate():
    pass

def main():
    if not os.path.exists("x_train.npy"):
        prepare_dataset()
    if not os.path.exists(model_name):
        train()
    
    evaluate()


if __name__ == "__main__":
    main()