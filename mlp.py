### Simple MLP based on the following tutorial: https://cambridgespark.com/content/tutorials/deep-learning-for-complete-beginners-recognising-handwritten-digits/index.html
### Changed into binary classifier by using sigmoid as activation function in output layer, and binary_crossentropy as loss function
### 
### Old data (1 frame per neg):
### 20 epochs gives around 92-95 % accuracy on test data and 100 % accuracy on recorded data
### 
### New data (multiple frames per neg):
### Acc (test, rec): 99.59, 98.63
### F-score (test, rec): 35.8, 0.56

import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.utils import np_utils
from pandas import read_pickle
import custom_metrics

# Read data
X_train, y_train = read_pickle("data/train.pickle")
X_test, y_test = read_pickle("data/test.pickle")
X_rec, y_rec = read_pickle("data/rec.pickle")

num_train = len(y_train)
num_test = len(y_test)
num_rec = len(y_rec)
height, width, depth = 89, 13, 1

X_train = X_train.reshape(num_train, height*width)
X_test = X_test.reshape(num_test, height*width)
X_rec = X_rec.reshape(num_rec, height*width)


'''
# For old pickles:
def preprocess(X):
    num = len(X)
    X = np.asarray(X)
    X = X.reshape(num, height*width)
    X += np.abs(X.min())
    X /= X.max()
    return X

X_train = preprocess(X_train)
X_test = preprocess(X_test)
X_rec = preprocess(X_rec)
'''

def train_model(X_train, y_train, path):
    # Define hyperparameters
    batch_size = 128
    num_epochs = 20
    hidden_size = 512

    # Define model
    inp = Input(shape=(height*width,))
    hidden_1 = Dense(hidden_size, activation='relu')(inp)
    hidden_2 = Dense(hidden_size, activation='relu')(hidden_1)
    out = Dense(1, activation='sigmoid')(hidden_2)
    model = Model(inputs=inp, outputs=out)

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy', custom_metrics.fmeasure])

    # Fit model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.1)

    # Save model
    model.save("models/mlp.h5")

    return model

train_model(X_train, y_train, "models/mlp.h5")


# Read model
model = load_model('models/mlp.h5', custom_objects={'fmeasure': custom_metrics.fmeasure})


# Evaluate model
print("Test data:")
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('f-score:', score[2])

print("Recorded data:")
score = model.evaluate(X_rec, y_rec, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('f-score:', score[2])