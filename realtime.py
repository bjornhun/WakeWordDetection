import pyaudio
import numpy as np
from matplotlib import pyplot as plt
from preprocessing import normalize
from python_speech_features import mfcc
from keras.models import load_model
import custom_metrics
import threading

CHUNKSIZE = 14400 # fixed chunk size

model = load_model('models/cnn.h5', custom_objects={'fmeasure': custom_metrics.fmeasure})

count=1

print("Listening...")

def rec():
    threading.Timer(0.10, rec).start()
    global count
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=CHUNKSIZE)
    data = stream.read(CHUNKSIZE)
    numpydata = np.fromstring(data, dtype=np.int16)
    X = normalize(mfcc(numpydata)).reshape(1, 89, 13, 1)
    if ([int(round(x[0])) for x in model.predict(X)] == [1]):
        print("Wake word detected #", count)
        count+=1
        ww_detected = 3
    stream.stop_stream()
    stream.close()
    p.terminate()

rec()