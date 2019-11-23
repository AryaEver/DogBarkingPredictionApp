import librosa
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import os
from numpy import genfromtxt
from scipy.io.wavfile import write
from pydub import AudioSegment

def shift(data):
        return np.roll(data, 1600)

def stretch(data, rate=1):
        input_length = 8820
        data = librosa.effects.time_stretch(data, rate)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data


path_mudi = '../DataBase/'
mudi2 = []
files_mudi = []
sec1_mudi = []
for item in os.listdir(path_mudi+'/1secUrban/'):
    files_mudi.append(item)
    print(len(files_mudi))
    sec, sr = librosa.load(path_mudi+'/1secUrban/'+item,sr=8820,duration=1.0)
    emptyspace = np.zeros(int(8820 - len(sec)))
    sec = np.concatenate((sec, emptyspace), axis=0)
    #add noise
    sec1 = sec + 0.05 * np.random.randn(len(sec))
    
    sec1 = sec1/abs(sec1).max()
    sec1_mudi.append(sec1)
    scaled = np.int16(sec1/np.max(np.abs(sec1)) * 32767)
    write('../DataBase/1secUrbanAugmentedNoise/noise0-05'+item, 8820, scaled)
    #if len(files_mudi) > 5:
    #    break

sec1_mudi = np.reshape(sec1_mudi,(len(files_mudi),len(sec1)))
csv_mudi = np.reshape(np.asarray(files_mudi),(len(files_mudi),1))
mudi = genfromtxt('./urban_labels.csv', delimiter=',',dtype='str')    
context_categories = mudi[:,1]
#individ_categories = mudi[:,2]
print('generating context file')
mudicontext = np.column_stack((sec1_mudi,context_categories))
np.savetxt('urban_augmented_raw_labels.csv', mudicontext, delimiter=',',fmt="%s")
del mudicontext
#print('generating individ file')
#mudiindivid = np.column_stack((sec1_mudi,individ_categories))
#np.savetxt('mescalina_individ_augmented_raw_labels.csv', mudiindivid, delimiter=',',fmt="%s")
