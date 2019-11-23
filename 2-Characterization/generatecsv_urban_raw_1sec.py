import pandas as pd
import librosa
import os
import numpy as np
from numpy import genfromtxt
from scipy.io.wavfile import write

mels = 128
path_urban = '../DataBase/UrbanSound8K'
urban2 = []
files_urban = []
sec1_urban = []
for item in os.listdir(path_urban+'/audio/'):
    files_urban.append(item)
    print(len(files_urban))
    sec1, sr = librosa.load(path_urban+'/audio/'+item,sr=8820,duration=1.0)
    emptyspace = np.zeros(int(8820 - len(sec1)))
    sec1 = np.concatenate((sec1, emptyspace), axis=0)
    sec1 = sec1/abs(sec1).max()
    sec1_urban.append(sec1)
    librosa.output.write_wav(path_urban+'/1sec/'+item, sec1, sr)
    scaled = np.int16(sec1/np.max(np.abs(sec1)) * 32767)
    write(path_urban+'/1sec/'+item, 8820, scaled)
    
    #if len(files_urban) > 49:
        #break

sec1_urban = np.reshape(sec1_urban,(len(files_urban),len(sec1)))
csv_urban = np.reshape(np.asarray(files_urban),(len(files_urban),1))
urban = genfromtxt(path_urban+'/UrbanSound8K.csv', delimiter=',',dtype='str')    
urban_categories = urban.T[7][1:]
urban_categories_num = pd.factorize(urban_categories)[0]
csv_urban = np.array([urban_categories,urban_categories_num],dtype=str).T
#csv_urban = csv_urban[:50]
csv_urban = np.concatenate((np.reshape(files_urban,(len(files_urban),1)),csv_urban), axis=1)
categories_vector = np.reshape(csv_urban[:,2],(len(csv_urban[:,2]),1))
categories_vector = pd.DataFrame(categories_vector)
sec1_urban = pd.DataFrame(sec1_urban)
sec1_urban = pd.concat([sec1_urban,categories_vector], axis=1)
sec1_urban = np.asarray(sec1_urban.values,dtype=float)
np.savetxt('urban.csv', csv_urban, delimiter=',',fmt="%s")
np.savetxt('urban_raw_1sec.csv', sec1_urban, delimiter=',',fmt="%s")
