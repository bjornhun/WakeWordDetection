import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc
import os

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
    Returns MFCC coefficients of the input signal as numpy array:
    [[[2D data], label], [[2D data], label], ... , [[2D data], label]]
    '''
    
    if len(x) < 14400:
        x = set_length(x)
    
    x_length = len(x)
    frames = []
    for i in range (0, x_length-frame_size+1, step_size):
        coeff = mfcc(x[i:(i+frame_size)])
        frames.append([coeff, label])
    return np.asarray(frames)

def preprocess():
    '''Reads all training data and gets MFCC coefficients.
    Returns np array with all MFCC data + label.
    '''
    filepath = data_path = os.getcwd() + "\\data\\train\\"
    os.chdir(filepath)
    files = os.listdir()
    num_of_files = len(files)

    data = []
    i = 1

    for f in files:
        if f.endswith("042.wav"):
            label = 1
        else:
            label = 0

        fs, x = wavfile.read(f)
        data.append(get_coefficients(x, label))
        print(str(i) + "/" + str(num_of_files) + " files preprocessed.")
        i+=1
    
    return np.asarray(data)


if __name__ == '__main__':
    #data = preprocess()
    #np.save("data.npy", data)