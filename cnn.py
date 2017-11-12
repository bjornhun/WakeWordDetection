### Simple CNN based on the following example: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
### Changed into binary classifier by using sigmoid as activation function in output layer, and binary_crossentropy as loss function
### 12 epochs and using test data as validation gives 97.16 % accuracy
### 12 epochs and using 10 % validation split gives 95.55 % accuracy

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from pandas import read_pickle

batch_size = 128
num_classes = 10
epochs = 12

# Read data
X_train, y_train = read_pickle("data/train.pickle")
X_test, y_test = read_pickle("data/test.pickle")

rows, columns = 89, 13

# Normalize
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
X_train += np.abs(X_train.min())
X_train /= X_train.max()
X_test += np.abs(X_test.min())
X_test /= X_test.max()
X_train = X_train.reshape(7776, 89, 13, 1)
X_test = X_test.reshape(2112, 89, 13, 1)

# Define model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(89, 13, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# Compile, fit, and evaluate
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.1)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])