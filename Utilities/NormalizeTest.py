import pandas as pd
import librosa
import os
import numpy as np
from numpy import genfromtxt
from scipy.io.wavfile import write
import matplotlib.pyplot as plt


path_mudi = '../DataBase/Mudidogs'
mudi2 = []
files_mudi = []
sec1_mudi = []
for item in os.listdir(path_mudi+'/rawaudio/'):
    files_mudi.append(item)
    print(len(files_mudi))
    sec1, sr = librosa.load(path_mudi+'/rawaudio/'+item,sr=8820,duration=1.0)
    emptyspace = np.zeros(int(8820 - len(sec1)))
    sec1 = np.concatenate((sec1, emptyspace), axis=0)
    sec1_mudi.append(sec1)
    scaled = np.int16(sec1/np.max(np.abs(sec1)) * 32767)
    write(path_mudi+'/1sec/'+item, 8820, scaled)
    if len(files_mudi) >= 500 :
        break

plt.subplot(211)
plt.plot(sec1)
normalized = sec1/abs(sec1).max()
plt.subplot(212)
plt.plot(normalized)