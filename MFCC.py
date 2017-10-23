from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

def uniformize(x, cutoff=80000):
    '''Gives segments appropriate length by either slicing or zero-padding.
    Default length/cutoff is 5 sec.'''
    num_samples = len(x)
    if num_samples > cutoff:
        return x[:cutoff]
    else:
        zeros = cutoff - num_samples
        return np.append(x, [0]*zeros)

def frame_partition(x, frame_size=400, step_size=160, cutoff=80000):
    '''Partitions sound signal into frames.
    Default is 25 msec frames with 10 msec steps.
    Returns a numpy array of 498 frames of length 400.'''
    frames = []
    for i in range (0, cutoff-frame_size, step_size):
        frames.append(x[i:(i+frame_size)])
    return np.asarray(frames)

def MFCC(fs, x):
    '''Returns a numpy array of 400x12 MFCC coefficients.'''
    x = uniformize(x, fs*5)
    frames = frame_partition(x)

if __name__ == '__main__':
    # Fs = 16000, samples = 53248
    fs, x = wavfile.read("data/train/f029_04_067.wav")
    MFCC(fs, x)
    #plt.plot(range(len(x)), x)
    #plt.show()
