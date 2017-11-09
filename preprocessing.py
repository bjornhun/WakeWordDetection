import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc
import os
from pandas import to_pickle, read_pickle

def set_length(x, cutoff=14400):
    '''Gives segments appropriate length by either slicing or zero-padding.
    Default length/cutoff is 900 msec.
    Returns input signal with modified length.'''

    num_samples = len(x)
    if num_samples > cutoff:
        return x[:cutoff]
    else:
        zeros = cutoff - num_samples
        return np.append(x, [0]*zeros)

def get_coefficients(x, frame_size=14400):
    '''Partitions sound signal into frames.
    Default is 900 msec frames.
    Frames start at first value over 5000 in order to encapsulate wakeword.
    Frames shorter than defined frame size get zero-padded.
    Calculates MFCC coefficients of the input signal.
    Returns MFCC data as 89x13 numpy array'''

    for i in range(len(x)):
        if x[i] > 5000:
            break
    
    if len(x[i:]) < frame_size:
        x = set_length(x, cutoff=(i+frame_size))

    return mfcc(x[i:i+frame_size])

def preprocess(filepath):
    '''Reads all data from filepath and gets MFCC coefficients.
    Returns combined list with MFCC data + label for all files.'''

    files = os.listdir(filepath)
    num_of_files = len(files)

    X = []
    y = []

    count = 1
    for f in files:
        if f.endswith("042.wav"):
            y.append(1)
        else:
            y.append(0)

        fs, x = wavfile.read(filepath + f)

        X.append(get_coefficients(x))

        print(str(count) + "/" + str(num_of_files) + " files preprocessed.")
        count+=1
    
    return X, y

def get_train_data():
    '''Reads and returns preprocessed training data from pickle if it exists.
    If not, preprocesses training data, writes to pickle, and returns.'''

    if os.path.isfile("data/train.pickle"):
        return read_pickle("data/train.pickle")
    X_train, y_train = preprocess("data/train/")
    to_pickle([X_train, y_train], "data/train.pickle")
    return [X_train, y_train]

def get_test_data():
    '''Reads and returns preprocessed test data from pickle if it exists.
    If not, preprocesses test data, writes to pickle, and returns.'''

    if os.path.isfile("data/test.pickle"):
        return read_pickle("data/test.pickle")
    X_test, y_test = preprocess("data/test/")
    to_pickle([X_test, y_test], "data/test.pickle")
    return [X_test, y_test]

if __name__ == '__main__':
    get_train_data()
    get_test_data()