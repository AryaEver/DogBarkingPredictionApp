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
sec3_urban = []
for item in os.listdir(path_urban+'/audio/'):
    files_urban.append(item)
    print(len(files_urban))
    sec3, sr = librosa.load(path_urban+'/audio/'+item,sr=8820,duration=3.0)
    emptyspace = np.zeros(int(8820*3 - len(sec3)))
    sec3 = np.concatenate((sec3, emptyspace), axis=0)
    sec3_urban.append(sec3)
    librosa.output.write_wav(path_urban+'/3sec/'+item, sec3, sr)
    scaled = np.int16(sec3/np.max(np.abs(sec3)) * 32767)
    write(path_urban+'/3sec/'+item, 8820, scaled)
    
    #if len(files_urban) > 49:
        #break
    
sec3_urban = np.reshape(sec3_urban,(len(files_urban),len(sec3)))
csv_urban = np.reshape(np.asarray(files_urban),(len(files_urban),1))
urban = genfromtxt(path_urban+'/UrbanSound8K.csv', delimiter=',',dtype='str')    
urban_categories = urban.T[7][1:]
urban_categories_num = pd.factorize(urban_categories)[0]
csv_urban = np.array([urban_categories,urban_categories_num],dtype=str).T
#csv_urban = csv_urban[:50]
csv_urban = np.concatenate((np.reshape(files_urban,(len(files_urban),1)),csv_urban), axis=1)
categories_vector = np.reshape(csv_urban[:,2],(len(csv_urban[:,2]),1))
categories_vector = pd.DataFrame(categories_vector)
sec3_urban = pd.DataFrame(sec3_urban)
sec3_urban = pd.concat([sec3_urban,categories_vector], axis=1)
sec3_urban = np.asarray(sec3_urban.values,dtype=float)
np.savetxt('urban.csv', csv_urban, delimiter=',',fmt="%s")
np.savetxt('urban_raw_3sec.csv', sec3_urban, delimiter=',',fmt="%s")
