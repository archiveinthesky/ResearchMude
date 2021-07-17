# -*- coding: utf-8 -*-
from numpy.core.defchararray import count
from pytube import YouTube
from pytube import Playlist
from moviepy.editor import *
from pydub import AudioSegment
import urllib.request
import time
import librosa
import os
import numpy as np

yt = YouTube("https://www.youtube.com/watch?v=MpKK0n0iyHU").streams.first().download(filename = "mother")
VideoFileClip("mother.mp4").audio.write_audiofile("jazz.mp3")
counter = 3000
editsound = AudioSegment.from_mp3("jazz.mp3")

#stddeviations = np.load("../stddevi.npy")
#average = np.load("../avrg.npy")

startfrom = 382

def splitmp3():
    global counter
    for i in range(88):
        print(i + startfrom)
        tmpcut = editsound[counter:counter + 30000]
        tmpcut.export("{}.mp3".format(i + startfrom), format = "mp3")
        counter += 36000

def transmfcc():
    global counter
    for i in range(88):
        print(i + startfrom)
        audio, sr = librosa.load("{}.mp3".format(i + startfrom))
        mfccaudio = np.reshape(librosa.feature.mfcc(audio), 25840)
        #formated = []
        #for j in range(25840):
        #    avrg = average[j]
        #    sigma = stddeviations[j]
        #    formated.append((mfccaudio[j] - avrg) / sigma)
        np.save("../j{}".format(i + startfrom), mfccaudio)

splitmp3()
transmfcc()
