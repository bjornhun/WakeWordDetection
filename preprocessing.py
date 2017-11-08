import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc

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

def get_coefficients(x, frame_size=14400, step_size=1600):
    '''Partitions sound signal into frames.
    Default is 900 msec frames with 100 msec steps.
    Returns MFCC coefficients of the input signal as 3D np array.'''
    
    if len(x) < 14400:
        x = set_length(x)
    
    x_length = len(x)
    frames = []
    for i in range (0, x_length-frame_size+1, step_size):
        coeff = mfcc(x[i:(i+frame_size)])
        frames.append(coeff)
    return np.asarray(frames)

if __name__ == '__main__':
    # Fs = 16000, samples = 53248
    '''fs, x = wavfile.read("data/train/f029_04_067.wav")
    coeff = mfcc(x)
    print(coeff.shape)'''

    for i in range(0, 1, 10):
        print(i)