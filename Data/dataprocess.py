import numpy as np
from numpy.core.defchararray import array
import pandas as pd
import librosa
import time

cafejazz = []
classicaljazz = []
lyrics = []

sr = 22050

old = time.time()

for i in range(470):
    cafejazz.append(librosa.load("Rawdata/CafeJazz/{}.mp3".format(i))[0])
    if i % 50 == 0:
        print(i)

print("Genre Done")

for i in range(470):
    classicaljazz.append(librosa.load("Rawdata/ClassicalJazz/{}.mp3".format(i))[0])
    if i % 50 == 0:
        print(i)

print("Genre Done")

for i in range(470):
    lyrics.append(librosa.load("Rawdata/Lyrics/{}.mp3".format(i))[0])
    if i % 50 == 0:
        print(i)

print("Genre Done, cost time {}".format(time.time() - old))
print("Starting Process")

def zerocrossingcalc(dataarr):
    result = []
    for each in dataarr:
        result.append(librosa.feature.zero_crossing_rate(each))
    return result

def spectralcentroid(dataarr):
    result = []
    for each in dataarr:
        result.append(librosa.feature.spectral_centroid(each, sr=sr))
    return result

def spectralrolloff(dataarr):
    result = []
    for each in dataarr:
        result.append(librosa.feature.spectral_rolloff(each, sr=sr))
    return result

def mfcc(dataarr):
    result = []
    for each in dataarr:
        result.append(librosa.feature.mfcc(each, sr=sr, n_mfcc=13))
    return result

def chromastft(dataarr):
    result = []
    for each in dataarr:
        result.append(librosa.feature.chroma_stft(each, sr=sr))
    return result

allzcc = []
allsc = []
allsr = []
allmfcc = []
allcsrft = []

for genre in [cafejazz, classicaljazz, lyrics]:
    
    for each in zerocrossingcalc(genre):
        allzcc.append(each)
    print("ZCC Done")
    
    
    for each in spectralcentroid(genre):
        allsc.append(each)
    print("SC Done")
    
    
    for each in spectralrolloff(genre):
        allsr.append(each)
    print("SR Done")
    

    for each in mfcc(genre):
        allmfcc.append(each)
    print("MFCC Done")
    
    
    for each in chromastft(genre):
        allcsrft.append(each)
    print("Csrft Done")
    
    print('Genre Done, cost time {}'.format(time.time() - old))

np.save("./Formated/zcc.npy", np.array(allzcc))
np.save("./Formated/sc.npy", np.array(allsc))
np.save("./Formated/sr.npy", np.array(allsr))
np.save("./Formated/mfcc.npy", np.array(allmfcc))
np.save("./Formated/csrft.npy", np.array(allcsrft))


"""savedataframe = pd.DataFrame(
    data=np.array([allzcc, allsc, allsr, allmfcc, allcsrft]),
    index=["ZCR", "SC", "SR", "MFCC", "CSRFT"],
    columns=[i for i in range(1410)]
)

savedataframe.to_csv("resultunoptimized.csv")"""