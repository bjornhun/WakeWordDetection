from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import np_utils
from pandas import read_pickle
import numpy as np

# Define hyperparameters
batch_size = 128
num_epochs = 20
hidden_size = 512

# Read data
X_train, y_train = read_pickle("data/train.pickle")
X_test, y_test = read_pickle("data/test.pickle")

num_train = len(y_train)
num_test = len(y_test)
height, width, depth = 89, 13, 1

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

# Reshape
X_train = X_train.reshape(num_train, height*width)
X_test = X_test.reshape(num_test, height*width)

# Normalize
X_train += np.abs(X_train.min())
X_train /= X_train.max()
X_test += np.abs(X_test.min())
X_test /= X_test.max()

# Define model
inp = Input(shape=(height*width,))
hidden_1 = Dense(hidden_size, activation='relu')(inp)
hidden_2 = Dense(hidden_size, activation='relu')(hidden_1)
out = Dense(1, activation='sigmoid')(hidden_2)
model = Model(inputs=inp, outputs=out)

# Compile model
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# Fit model
model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.1)

# Evaluate model
scores = model.evaluate(X_test, y_test, verbose=1)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))