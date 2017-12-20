import pyaudio
import numpy as np
from preprocessing import normalize
from python_speech_features import mfcc
from keras.models import load_model
import threading
import winsound

# Filename of model
modelname = "cnn2_100.h5"
# Number of consequtive positives required for wake word detection
min_positives=3

model = load_model('models/' + modelname)

count = 1
buff = [0]*10
CHUNKSIZE = 16000

print("Listening...")

def detectWakewords():
    threading.Timer(0.10, detectWakewords).start()
    global count
    global buff
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=CHUNKSIZE)
    data = stream.read(CHUNKSIZE)
    raw_speech = np.fromstring(data, dtype=np.int16)
    if modelname.startswith('cnn'):
        X = normalize(mfcc(raw_speech)).reshape(1, 99, 13, 1)
    else:
        X = normalize(mfcc(raw_speech)).reshape(1, 99 * 13)
    pred = [int(round(x[0])) for x in model.predict(X)]
    if np.sum(np.abs(raw_speech)) < len(buff)*100000:
        pred = [0]
    buff.pop(0)
    buff.append(pred[0])
    if (buff[-min_positives:] == [1]*min_positives and buff[-(min_positives+1)] == 0):
        print("Wake word detected #", count)
        winsound.Beep(1000, 300)
        count+=1        
    stream.stop_stream()
    stream.close()
    p.terminate()

detectWakewords()