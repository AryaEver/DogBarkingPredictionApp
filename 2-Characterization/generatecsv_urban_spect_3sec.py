import pandas as pd
import librosa
import os
import numpy as np
from numpy import genfromtxt

mels = 128
path_urban = '../DataBase/UrbanSound8K'
urban2 = []
files_urban = []
spect3_urban = []
for item in os.listdir(path_urban+'/audio/'):
    files_urban.append(item)
    print(len(files_urban))
    sec3, sr = librosa.load(path_urban+'/audio/'+item,sr=8820,duration=3.0)
    emptyspace = np.zeros(int(8820*3 - len(sec3)))
    sec3 = np.concatenate((sec3, emptyspace), axis=0)
    spect3 = librosa.feature.melspectrogram(y=sec3,sr=sr,n_mels=mels)
    spect3 = np.reshape(spect3,(len(spect3[:])*len(spect3[1]),1))
    spect3_urban.append(spect3)
    
    #if len(files_urban) > 49:
        #break
        
spect3_urban = np.reshape(spect3_urban,(len(files_urban),len(spect3)))
csv_urban = np.reshape(np.asarray(files_urban),(len(files_urban),1))
urban = genfromtxt(path_urban+'/UrbanSound8K.csv', delimiter=',',dtype='str')    
urban_categories = urban.T[7][1:]
urban_categories_num = pd.factorize(urban_categories)[0]
csv_urban = np.array([urban_categories,urban_categories_num],dtype=str).T
#csv_urban = csv_urban[:50]
csv_urban = np.concatenate((np.reshape(files_urban,(len(files_urban),1)),csv_urban), axis=1)
categories_vector = np.reshape(csv_urban[:,2],(len(csv_urban[:,2]),1))

csv_urban_spect_3sec = np.concatenate((spect3_urban,categories_vector), axis=1)
np.savetxt('urban.csv', csv_urban, delimiter=',',fmt="%s")
np.savetxt('urban_spect_3sec.csv', csv_urban_spect_3sec, delimiter=',',fmt="%s")