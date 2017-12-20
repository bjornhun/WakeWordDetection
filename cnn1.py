# Simple CNN based on the following example:
# https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from pandas import read_pickle
from sklearn.metrics import confusion_matrix

batch_size = 128

X_train, y_train = read_pickle("data/train.pickle")
X_test, y_test = read_pickle("data/test.pickle")

(num_train, rows, columns) = X_train.shape
num_test = X_test.shape[0]

X_train = X_train.reshape(num_train, rows, columns, 1)
X_test = X_test.reshape(num_test, rows, columns, 1)

def train_model(X_train, y_train, path):
    model = Sequential()
    model.add(Conv2D(32,
                    kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(rows, columns, 1)))
    model.add(Conv2D(64, 
                    (3, 3),
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, 
                    activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,
                    activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                    optimizer='adam',
                    metrics=['accuracy'])
    model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
    model.save(path)

    return model


epochs = 10
filename = "cnn1_10.h5"
train_model(X_train, y_train, "models/" + filename)

epochs = 50
filename = "cnn1_50.h5"
train_model(X_train, y_train, "models/" + filename)

epochs = 100
filename = "cnn1_100.h5"
train_model(X_train, y_train, "models/" + filename)