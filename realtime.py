import pyaudio
import numpy as np
from matplotlib import pyplot as plt
from preprocessing import normalize
from python_speech_features import mfcc
from keras.models import load_model
import custom_metrics
import threading
import winsound

CHUNKSIZE = 16000 # fixed chunk size

model = load_model('models/cnn2_100.h5')

count=1
buff = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
predbuff=[0,0,0,0,0]

print("Listening...")

def rec():
    threading.Timer(0.10, rec).start()
    global count
    global buff
    global predbuff
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=CHUNKSIZE)
    data = stream.read(CHUNKSIZE)
    numpydata = np.fromstring(data, dtype=np.int16)
    X = normalize(mfcc(numpydata)).reshape(1, 99, 13, 1)
    pred = [int(round(x[0])) for x in model.predict(X)]
    if np.sum(np.abs(numpydata)) < len(buff)*100000:
        pred = [0]
    buff.pop(0)
    buff.append(pred[0])
    #predbuff.pop(0)
    #predbuff.append(model.predict(X)[0])
    if (buff[-3:] == [1,1,1] and buff[-4] == 0):
        print("Wake word detected #", count)
#        print(buff)
#        print(predbuff)
        winsound.Beep(1000, 300)
        count+=1        
    
    '''if ([int(round(x[0])) for x in model.predict(X)] == [1]):
        print("Wake word detected #", count)
        count+=1'''
    stream.stop_stream()
    stream.close()
    p.terminate()

rec()