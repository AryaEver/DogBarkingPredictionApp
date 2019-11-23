import pandas as pd
import librosa
import os
import numpy as np
from numpy import genfromtxt

mels = 128
path_urban = '../DataBase/UrbanSound8K'
urban2 = []
files_urban = []
spect1_urban = []
for item in os.listdir(path_urban+'/audio/'):
    files_urban.append(item)
    print(len(files_urban))    
    sec1, sr = librosa.load(path_urban+'/audio/'+item,sr=8820,duration=1.0)
    emptyspace = np.zeros(int(8820 - len(sec1)))
    sec1 = np.concatenate((sec1, emptyspace), axis=0)
    spect1 = librosa.feature.melspectrogram(y=sec1,sr=sr,n_mels=mels)
    spect1 = np.reshape(spect1,(len(spect1[:])*len(spect1[1]),1))
    spect1_urban.append(spect1)
    
    if len(files_urban) > 49:
        break
    
spect1_urban = np.reshape(spect1_urban,(len(files_urban),len(spect1)))
csv_urban = np.reshape(np.asarray(files_urban),(len(files_urban),1))
urban = genfromtxt(path_urban+'/UrbanSound8K.csv', delimiter=',',dtype='str')    
urban_categories = urban.T[7][1:]
urban_categories_num = pd.factorize(urban_categories)[0]
csv_urban = np.array([urban_categories,urban_categories_num],dtype=str).T
#csv_urban = csv_urban[:50]
csv_urban = np.concatenate((np.reshape(files_urban,(len(files_urban),1)),csv_urban), axis=1)
categories_vector = np.reshape(csv_urban[:,2],(len(csv_urban[:,2]),1))

csv_urban_spect_1sec = np.concatenate((spect1_urban,categories_vector), axis=1)
np.savetxt('urban.csv', csv_urban, delimiter=',',fmt="%s")
np.savetxt('urban_spect_1sec.csv', csv_urban_spect_1sec, delimiter=',',fmt="%s")