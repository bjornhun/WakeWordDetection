import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
import os
from pandas import to_pickle, read_pickle
from scipy.signal import medfilt

def set_length(x, cutoff):
    '''Gives segments appropriate length by either slicing or zero-padding.
    Returns input signal with modified length.'''
    num_samples = len(x)
    if num_samples > cutoff:
        return x[:cutoff]
    else:
        zeros = cutoff - num_samples
        return np.append(x, [0]*zeros)

def normalize(coeff):
    '''Normalizes numpy array of coefficients'''
    coeff += np.abs(coeff.min())
    coeff /= coeff.max()
    return coeff

def get_coefficients(x, frame_size=16000, threshold=1000, jump=0):
    '''Encapsulates speech in frame of given size, and finds MFCC coefficients.
    Threshold is used to detect speech.
    Jump is used in order to get different frames from the same signal.
    Returns MFCC data as 99x13 numpy array'''
    # Median filter to remove shot noise
    med = medfilt(x, 15)                
    for i in range(1000, len(med)):
        if med[i] > threshold:
            break
    i -= (600 - jump)
    i = max(i,0)
    x = set_length(x[i:], 16000)
    coeff = normalize(mfcc(x))
    return coeff

def preprocess(filepath):
    '''Reads all data from filepath and gets MFCC coefficients.
    Returns MFCC data and corresponding labels for all files.'''
    files = os.listdir(filepath)
    num_of_files = len(files)
    X = []
    y = []
    count = 1
    for f in files:
        fs, x = wavfile.read(filepath + f)
        if f.endswith("042.wav"):
            X.append(get_coefficients(x))
            y.append(1)
            X.append(get_coefficients(x, jump=100))
            y.append(1)
            X.append(get_coefficients(x, jump=200))
            y.append(1)
        else:
            X.append(get_coefficients(x))
            y.append(0)
        print(str(count) + "/" + str(num_of_files) + " files preprocessed.")
        count+=1
    return np.asarray(X), y

def get_train_data():
    '''Reads and returns preprocessed training data from pickle if it exists.
    If not, preprocesses training data, writes to pickle, and returns.'''
    if os.path.isfile("data/train.pickle"):
        return read_pickle("data/train.pickle")
    X_train, y_train = preprocess("data/train/")
    to_pickle((X_train, y_train), "data/train.pickle")
    return (X_train, y_train)

def get_test_data():
    '''Reads and returns preprocessed test data from pickle if it exists.
    If not, preprocesses test data, writes to pickle, and returns.'''
    if os.path.isfile("data/test.pickle"):
        return read_pickle("data/test.pickle")
    X_test, y_test = preprocess("data/test/")
    to_pickle((X_test, y_test), "data/test.pickle")
    return (X_test, y_test)

if __name__ == '__main__':
    get_train_data()
    get_test_data()