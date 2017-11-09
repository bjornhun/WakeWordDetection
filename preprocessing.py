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

def get_coefficients(x, label, frame_size=14400, step_size=1600):
    '''Partitions sound signal into frames.
    Default is 900 msec frames with 100 msec steps.
    Calculates MFCC coefficients of the input signal.
    Returns a list on the following format, where the MFCC data is a 2D np array:
    [[[2D data], label], [[2D data], label], ... , [[2D data], label]]'''
    
    if len(x) < 14400:
        x = set_length(x)
    
    x_length = len(x)
    frames = []
    for i in range (0, x_length-frame_size+1, step_size):
        coeff = mfcc(x[i:(i+frame_size)])
        frames.append([coeff, label])
    return frames

def preprocess(filepath):
    '''Reads all data from filepath and gets MFCC coefficients.
    Returns combined list with MFCC data + label for all files.'''

    files = os.listdir(filepath)
    num_of_files = len(files)

    data = []
    i = 1

    for f in files:
        if f.endswith("042.wav"):
            label = 1
        else:
            label = 0

        fs, x = wavfile.read(filepath + f)
        data.extend(get_coefficients(x, label))
        print(str(i) + "/" + str(num_of_files) + " files preprocessed.")
        i+=1
    
    return data

def get_train_data():
    '''Reads and returns preprocessed training data from pickle if it exists.
    If not, preprocesses training data, writes to pickle, and returns.'''

    if os.path.isfile("data/train.pickle"):
        return read_pickle("data/train.pickle")
    data = preprocess("data/train/")
    to_pickle(data, "data/train.pickle")
    return data

def get_test_data():
    '''Reads and returns preprocessed test data from pickle if it exists.
    If not, preprocesses test data, writes to pickle, and returns.'''

    if os.path.isfile("data/test.pickle"):
        return read_pickle("data/test.pickle")
    data = preprocess("data/test/")
    to_pickle(data, "data/test.pickle")
    return data

if __name__ == '__main__':
    get_train_data()
    get_test_data()