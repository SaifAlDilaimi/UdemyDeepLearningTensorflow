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
    lr = learning_rate
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

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

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
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
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape,)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = Add()([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    
    opt_ada = Adam(lr=lr_schedule(0))
    model.compile(loss=categorical_crossentropy, optimizer=opt_ada, metrics=['accuracy'])

    return model

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