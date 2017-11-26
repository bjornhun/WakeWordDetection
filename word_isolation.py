import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import medfilt, argrelmin
import os

def split_words(path):
    print(path)
    fs, data = wavfile.read(path)
    abs_data = [abs(x) for x in data]
    med_data = medfilt(abs_data, 15)
    avg_data = np.convolve(med_data, np.ones((1000,))/1000, mode='valid')

    i = np.argmax(np.asarray(med_data[:int(len(data)/2)]))

    minima = argrelmin(avg_data, order=1000)
    minima = [x for x in minima[0] if x>i]

    try:
        div = minima[0]

    except:
        print(minima)
        plot_speech(path)

    w1 = data[:div]
    w2 = data[div:]

    wavfile.write(path[:-4] + "_door.wav", fs, w1)
    wavfile.write(path[:-4] + "_open.wav", fs, w2)

def plot_speech(path):
    fs, x = wavfile.read(path)
    plt.plot(range(len(x)), x)
    plt.show()

def split_path(path):
    files = os.listdir(path)
    for f in files:
        if f.endswith("042.wav"):
            split_words(path + f)

if __name__ == '__main__':
    split_path("data_ws/train/")
    split_path("data_ws/test/")