import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy

import numpy as np
import matplotlib.pyplot as plt

learning_rate = 1e-3
batch_size = 256
epochs = 15
num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral

def prepare_dataset():
    with open("fer2013\\fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)

    num_of_instances = lines.size
    print("Anzahl der Zeilen: ",num_of_instances)

    # trainset and test set
    x_train, y_train, x_test, y_test = [], [], [], []

    #transfer train and test set data
    for i in range(1, num_of_instances):
        try:
            emotion, img, usage = lines[i].split(",")
            
            img_values = img.split(" ")
                
            pixels = np.array(img_values, 'float32')
            
            emotion = to_categorical(emotion, num_classes)
        
            if 'Training' in usage:
                x_train.append(pixels)
                y_train.append(emotion)
            elif 'PublicTest' in usage:
                x_test.append(pixels)
                y_test.append(emotion)
        except:
            print("",end="")


    # data transformation für train and test sets
    x_train = np.array(x_train, 'float32')
    y_train = np.array(y_train, 'float32')
    x_test = np.array(x_test, 'float32')
    y_test = np.array(y_test, 'float32')

    x_train /= 255 #normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    x_test = x_test.astype('float32')

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    np.save("x_train", x_train)
    np.save("y_train", y_train)
    np.save("x_test", x_test)
    np.save("y_test", y_test)

def train():
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    x_test =  np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    model = Sequential()

    #1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

    #2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    #3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    model.add(Flatten())

    #fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    opt = RMSprop(lr=learning_rate)
    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    model.save('expression_predictor')

def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()

def evaluate():
    x_test =  np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    model = load_model('expression_predictor')
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', 100*score[1])

    predictions = model.predict(x_test)
    
    index = 0
    for i in predictions:
        if index < 30 and index >= 20:
            testing_img = np.array(x_test[index], 'float32')
            testing_img = testing_img.reshape([48, 48]);
            
            plt.gray()
            plt.imshow(testing_img)
            plt.show()
            
            print(i)
            
            emotion_analysis(i)
            
            print("----------------------------------------------")
        index = index + 1

def main():
    if not os.path.exists("x_train.npy"):
        prepare_dataset()
    if not os.path.exists("expression_predictor"):
        train()
    
    evaluate()


if __name__ == "__main__":
    main()