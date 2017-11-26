### Simple CNN based on the following example: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
### Changed into binary classifier by using sigmoid as activation function in output layer, and binary_crossentropy as loss function
### 12 epochs gives 97 % accuracy on test data and 50-66 % accuracy on recorded data
###
### New new data (random frame per neg):
### Acc (test): 98.1
### F-score (test): 0.962
###
### Tried without improvement:
### Zero-padding (padding='same')
### L2 regularization with lambda=0.0001

import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from pandas import read_pickle
import custom_metrics

batch_size = 128
num_classes = 3
epochs = 12

# Read data
X_train, y_train = read_pickle("data_ws/train.pickle")
X_test, y_test = read_pickle("data_ws/test.pickle")

'''num_neg = len([x for x in y_train if x==0])/len(y_train)
num_pos = len([x for x in y_train if x==1])/len(y_train)

class_weight = {0:num_pos, 1:num_neg}'''

'''
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
'''

(num_train, rows, columns) = X_train.shape
num_test = X_test.shape[0]

'''
# Normalize
for coeff in X_train:
    coeff += np.abs(coeff.min())
    coeff /= coeff.max()

for coeff in X_test:
    coeff += np.abs(coeff.min())
    coeff /= coeff.max()
'''

X_train = X_train.reshape(num_train, rows, columns, 1)
X_test = X_test.reshape(num_test, rows, columns, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def train_model(X_train, y_train, path):
    # Define model
    model = Sequential()
    model.add(Conv2D(32,
                    kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(rows, columns, 1),
                    kernel_initializer='he_uniform'))
    model.add(Conv2D(32, 
                    (3, 3),
                    activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, 
                    activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(3,
                    activation='softmax',
                    kernel_initializer='glorot_uniform'))

    # Compile, fit, and save
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer='adam',
                metrics=['accuracy', custom_metrics.fmeasure])

    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=0.1)

    # Save model
    model.save(path)

    return model

train_model(X_train, y_train, "models/cnn_ws.h5")

# Read model
model = load_model('models/cnn_ws.h5', custom_objects={'fmeasure': custom_metrics.fmeasure})
          
# Evaluate model
print("Test data:")
score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('f-score:', score[2])