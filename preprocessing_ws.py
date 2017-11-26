import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc
import os
from pandas import to_pickle, read_pickle
from random import randint

def set_length(x, placement="start", cutoff=16000):
    '''Gives segments appropriate length by either slicing or zero-padding.
    Default length/cutoff is 900 msec.
    Returns input signal with modified length.'''

    num_samples = len(x)
    if num_samples > cutoff:
        if placement == "end":
            return x[-cutoff:]
        return x[:cutoff]
    else:
        zeros = cutoff - num_samples
        return np.append(x, [0]*zeros)

def normalize(coeff):
    'Normalizes numpy array of coefficients'
    coeff += np.abs(coeff.min())
    coeff /= coeff.max()
    return coeff

def get_coefficients(x, label, frame_size=16000, step_size=1600):
    '''Partitions sound signal into frames.
    Default is 900 msec frames.
    Frames start at first value over 5000 in order to encapsulate wakeword.
    Frames shorter than defined frame size get zero-padded.
    Calculates MFCC coefficients of the input signal.
    Returns MFCC data as 89x13 numpy array'''

    if label == 0:
        x = set_length(x, placement="end")
        coeff = mfcc(x)
        coeff = normalize(coeff)
        return coeff

    elif label == 1:
        x = set_length(x)
        coeff = mfcc(x)
        coeff = normalize(coeff)
        return coeff

    elif label == 3:
        # noisy start
        for i in range(3000, len(x)):
            if x[i] > 5000:
                break
        
        # ensure whole wakeword is included
        i -= 1600

        if len(x[i:]) < frame_size:
            x = set_length(x, cutoff=(i+frame_size))

        coeff = mfcc(x[i:i+frame_size])
        coeff = normalize(coeff)

        return coeff

    else:
        frames = []
        x_length = len(x)

        if x_length < frame_size:
            x = set_length(x)
            x_length = len(x)

        for i in range (0, x_length-frame_size+1, step_size):
            coeff = mfcc(x[i:(i+frame_size)])
            coeff = normalize(coeff)
            frames.append(coeff)
        i = randint(0, len(frames)-1)

        return frames[int(len(frames)/2)]

def preprocess(filepath):
    '''Reads all data from filepath and gets MFCC coefficients.
    Returns combined list with MFCC data + label for all files.'''

    files = os.listdir(filepath)
    num_of_files = len(files)

    X = []
    y = []

    count = 1
    neg = 1
    for f in files:
        fs, x = wavfile.read(filepath + f)

        if f.endswith("042.wav"):
            X.append(get_coefficients(x, 3))
            y.append(1)
        if (f.endswith("door.wav")):
            X.append(get_coefficients(x, 0))
            y.append(0)
        elif (f.endswith("open.wav")):
            X.append(get_coefficients(x, 1))
            y.append(0)
        else:
            if neg % 4 != 0:
                neg += 1
                continue
            X.append(get_coefficients(x, 2))
            y.append(0)
            neg += 1
        print(str(count) + "/" + str(num_of_files) + " files preprocessed.")
        count+=1
    
    return np.asarray(X), y

def get_train_data():
    '''Reads and returns preprocessed training data from pickle if it exists.
    If not, preprocesses training data, writes to pickle, and returns.'''

    if os.path.isfile("data_ws/train.pickle"):
        return read_pickle("data_ws/train.pickle")
    X_train, y_train = preprocess("data_ws/train/")
    to_pickle((X_train, y_train), "data_ws/train.pickle")
    return (X_train, y_train)

def get_test_data():
    '''Reads and returns preprocessed test data from pickle if it exists.
    If not, preprocesses test data, writes to pickle, and returns.'''

    if os.path.isfile("data_ws/test.pickle"):
        return read_pickle("data_ws/test.pickle")
    X_test, y_test = preprocess("data_ws/test/")
    to_pickle((X_test, y_test), "data_ws/test.pickle")
    return (X_test, y_test)

if __name__ == '__main__':
    get_train_data()
    get_test_data()