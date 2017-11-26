import pyaudio
import numpy as np
from matplotlib import pyplot as plt
from preprocessing import normalize
from python_speech_features import mfcc
from keras.models import load_model
import custom_metrics
import threading

CHUNKSIZE = 8000 # fixed chunk size

model = load_model('models/cnn_ws.h5', custom_objects={'fmeasure': custom_metrics.fmeasure})

count=1
buff = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

print("Listening...")

def rec():
    threading.Timer(0.10, rec).start()
    global count
    global buff
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=CHUNKSIZE)
    data = stream.read(CHUNKSIZE)
    numpydata = np.fromstring(data, dtype=np.int16)
    X = normalize(mfcc(numpydata)).reshape(1, 49, 13, 1)
    y_prob = model.predict(X) 
    y_class = y_prob.argmax(axis=-1)    
    if np.sum(np.abs(numpydata)) < 10000000:
        y_class = [2]
    buff.pop(0)
    buff.append(y_class[0])
    if (buff[-1] == 1 and 0 in buff):
        print(buff)
        print("Wake word detected #", count)
        count+=1        
    
    '''if ([int(round(x[0])) for x in model.predict(X)] == [1]):
        print("Wake word detected #", count)
        count+=1'''
    stream.stop_stream()
    stream.close()
    p.terminate()

rec()