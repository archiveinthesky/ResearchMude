import librosa
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras

stddeviations = np.load("./stddevi.npy")
average = np.load("./avrg.npy")

model = keras.models.load_model('../')

print("Ready to go!")

res = []
for i in range(530):
    audiopath = "./lyrics/{}.mp3".format(i + 470)
    try:
        audio, sr = librosa.load(audiopath)
        mfccaudio = np.reshape(librosa.feature.mfcc(audio, n_mfcc=13), 16796)
        formated = []
        for j in range(16796):
            avrg = average[j]
            sigma = stddeviations[j]
            formated.append((mfccaudio[j] - avrg) / sigma)
        res.append(formated)
        print(i)
    except:
        pass

res = np.array(res)
np.save("./Lyricsnpy.npy", res)

ans = []

for i in range(len(res)):
    ans.append([0,0,1])
ans = np.array(ans)

testacc = model.evaluate(res, ans)
print(testacc)