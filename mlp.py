# Simple MLP based on the following tutorial:
# https://cambridgespark.com/content/tutorials/deep-learning-for-complete-beginners-recognising-handwritten-digits/index.html

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Input, Dense
from pandas import read_pickle
from sklearn.metrics import confusion_matrix

batch_size = 128

X_train, y_train = read_pickle("data/train.pickle")
X_test, y_test = read_pickle("data/test.pickle")

(num_train, rows, columns) = X_train.shape
num_test = X_test.shape[0]

X_train = X_train.reshape(num_train, rows * columns)
X_test = X_test.reshape(num_test, rows * columns)

def train_model(X_train, y_train, path):
    model = Sequential()
    model.add(Dense(512,
                    activation='relu',
                    input_shape=(rows * columns,)))
    model.add(Dense(512,
                    activation='relu'))
    model.add(Dense(1,
                    activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer = 'adam',
                    metrics=['accuracy'])
    model.fit(X_train, y_train,
                    batch_size=batch_size,  
                    epochs=epochs, verbose=1,
                    validation_split=0.1)
    model.save("models/" + filename)

    return model


epochs = 10
filename = "mlp10.h5"
train_model(X_train, y_train, "models/" + filename)

epochs = 50
filename = "mlp50.h5"
train_model(X_train, y_train, "models/" + filename)

epochs = 100
filename = "mlp100.h5"
train_model(X_train, y_train, "models/" + filename)