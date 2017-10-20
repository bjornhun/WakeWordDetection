from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

def uniformize(x, cutoff=80000):
    '''Give segments appropriate length by either cutting of zero-padding.
    Default is 5 sec cutoff.'''
    num_samples = len(x)
    if num_samples > cutoff:
        return x[:cutoff]
    else:
        zeros = cutoff - num_samples
        return np.append(x, [0]*zeros)

def frame_partition(x, frame_size=400, step_size=160, cutoff=80000):
    '''Partition sound signal into frames.
    Default is 25 msec frames with 10 msec steps.'''
    frames = np.array()
    for i in range (0, cutoff, step_size):
        np.append(frames, x[i:(i+frame_size)])
    return frames

def MFCC(fs, x):
    #x = uniformize(x, fs*5)
    pass

if __name__ == '__main__':
    # Fs = 16000, samples = 53248
    fs, x = wavfile.read("data/train/f029_01_021.wav")
    plt.plot(range(len(x)), x)
    plt.show()
    #MFCC(fs, x)


# 25 ms frames, 10 ms steps, zero padding for last one

# each frame -> 12 MFCC coefficients
