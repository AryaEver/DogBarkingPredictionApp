import pandas as pd
import librosa
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import csv
import wave
from scipy.io.wavfile import write

'''
path_mudi = '../DataBase/Mudidogs'
files_mudi = os.listdir(path_mudi+'/rawaudio')
csv_mudi = np.reshape(np.asarray(files_mudi),(len(files_mudi),1))
mudi = genfromtxt(path_mudi+'/mudidogs.csv', delimiter=',',dtype='str')
mudi_individuos = mudi.T[1][1:]
mudi_individuos_num = pd.factorize(mudi_individuos)[0]
mudi_contexto = mudi.T[3][1:]
mudi_contexto_num = pd.factorize(mudi_contexto)[0]
mudi = np.array([mudi_individuos,mudi_individuos_num,mudi_contexto,mudi_contexto_num],dtype=str)
csv_mudi = np.concatenate((np.asarray(csv_mudi),mudi.T),axis=1)
np.savetxt('mudidogs.csv', csv_mudi, delimiter=',',fmt="%s")
'''

path_mescalina = '../DataBase/Mescalina'
files_mescalina = []
mescalina_individuos = []
mescalina_contexto = []
sec1_mescalina = []
for item in os.listdir(path_mescalina+'/'):
    for subitem in os.listdir(path_mescalina+'/'+item):       
        for subsubitem in os.listdir(path_mescalina+'/'+item+'/'+subitem):
            if 'L-S' == subitem:
                continue
            elif 'L-W' == subitem:
                continue
            elif 'L' in subitem:
                mescalina_individuos.append(item)
                mescalina_contexto.append(subitem)               
                files_mescalina.append(subsubitem)
                print(len(files_mescalina))
                sec1, sr = librosa.load(path_mescalina+'/'+item+'/'+subitem+'/'+subsubitem,sr=8820,duration=1.0)
                emptyspace = np.zeros(int(8820 - len(sec1)))
                sec1 = np.concatenate((sec1, emptyspace), axis=0)
                sec1 = sec1/abs(sec1).max()
                sec1_mescalina.append(sec1)
                scaled = np.int16(sec1/np.max(np.abs(sec1)) * 32767)
                write('../DataBase/Mescalina1sec/'+item+'_'+subitem+'_'+subsubitem, 8820, scaled)

                
mescalina_individuos_num = pd.factorize(mescalina_individuos)
print(mescalina_individuos_num[1])
mescalina_contexto_num = pd.factorize(mescalina_contexto)
print(mescalina_contexto_num[1])    
mescalina_individuos = np.reshape(np.asarray(mescalina_individuos),(len(mescalina_individuos),1))
mescalina_individuos_num = np.reshape(np.asarray(mescalina_individuos_num[0]),(len(mescalina_individuos_num[0]),1))
mescalina_contexto = np.reshape(np.asarray(mescalina_contexto),(len(mescalina_contexto),1))
mescalina_contexto_num = np.reshape(np.asarray(mescalina_contexto_num[0]),(len(mescalina_contexto_num[0]),1))
files_mescalina = np.reshape(np.asarray(files_mescalina),(len(files_mescalina),1))
csv_mescalina = np.concatenate((files_mescalina,
                                mescalina_individuos,mescalina_individuos_num,
                                mescalina_contexto,mescalina_contexto_num), axis=1)
sec1_mescalina = np.reshape(sec1_mescalina,(len(files_mescalina),len(sec1)))
#csv_mescalina = np.reshape(np.asarray(files_mescalina),(len(files_mescalina),1))
np.savetxt('mescalina.csv', csv_mescalina, delimiter=',',fmt="%s")
mescalina = genfromtxt('./mescalina.csv', delimiter=',',dtype='str')    
context_categories = csv_mescalina[:,4]
individ_categories = csv_mescalina[:,2]
mescalinacontext = np.column_stack((sec1_mescalina,context_categories))
np.savetxt('mescalina_context_raw.csv', mescalinacontext, delimiter=',',fmt="%s")
del mescalinacontext
print('generating individ file')
mescalinaindivid = np.column_stack((sec1_mescalina,individ_categories))
np.savetxt('mescalina_individ_raw.csv', mescalinaindivid, delimiter=',',fmt="%s")
del mescalinaindivid

'''
mels = 128
path_urban = '../DataBase/UrbanSound8K'
urban2 = []
files_urban = []
sec3_urban = []
spect3_urban = []
sec1_urban = []
spect1_urban = []
for item in os.listdir(path_urban+'/audio/'):
    print(len(files_urban))
    files_urban.append(item)
    sec3, sr = librosa.load(path_urban+'/audio/'+item,sr=22050,duration=3.0)
    emptyspace = np.zeros(int(22050*3 - len(sec3)))
    sec3 = np.concatenate((sec3, emptyspace), axis=0)
    sec3_urban.append(sec3)
    spect3 = librosa.feature.melspectrogram(y=sec3,sr=sr,n_mels=mels)
    spect3 = np.reshape(spect3,(len(spect3[:])*len(spect3[1]),1))
    spect3_urban.append(spect3)
    librosa.output.write_wav(path_urban+'/3sec/'+item, sec3, sr)
    
    sec1, sr = librosa.load(path_urban+'/audio/'+item,sr=22050,duration=1.0)
    emptyspace = np.zeros(int(22050 - len(sec1)))
    sec1 = np.concatenate((sec1, emptyspace), axis=0)
    sec1_urban.append(sec1)
    spect1 = librosa.feature.melspectrogram(y=sec1,sr=sr,n_mels=mels)
    spect1 = np.reshape(spect1,(len(spect1[:])*len(spect1[1]),1))
    spect1_urban.append(spect1)
    librosa.output.write_wav(path_urban+'/1sec/'+item, sec1, sr)
    
    #if len(files_urban) > 49:
        #break
    
sec3_urban = np.reshape(sec3_urban,(len(files_urban),len(sec3)))
sec1_urban = np.reshape(sec1_urban,(len(files_urban),len(sec1)))
spect3_urban = np.reshape(spect3_urban,(len(files_urban),len(spect3)))
spect1_urban = np.reshape(spect1_urban,(len(files_urban),len(spect1)))
csv_urban = np.reshape(np.asarray(files_urban),(len(files_urban),1))
urban = genfromtxt(path_urban+'/UrbanSound8K.csv', delimiter=',',dtype='str')    
urban_categories = urban.T[7][1:]
urban_categories_num = pd.factorize(urban_categories)[0]
csv_urban = np.array([urban_categories,urban_categories_num],dtype=str).T
#csv_urban = csv_urban[:50]
csv_urban = np.concatenate((np.reshape(files_urban,(len(files_urban),1)),csv_urban), axis=1)
categories_vector = np.reshape(csv_urban[:,2],(len(csv_urban[:,2]),1))
csv_urban_raw_1sec = np.concatenate((sec1_urban,categories_vector), axis=1)
csv_urban_raw_3sec = np.concatenate((sec3_urban,categories_vector), axis=1)

csv_urban_spect_1sec = np.concatenate((spect1_urban,categories_vector), axis=1)
csv_urban_spect_3sec = np.concatenate((spect3_urban,categories_vector), axis=1)
np.savetxt('urban.csv', csv_urban, delimiter=',',fmt="%s")
np.savetxt('urban_raw_1sec.csv', csv_urban_raw_1sec, delimiter=',',fmt="%s")
np.savetxt('urban_raw_3sec.csv', csv_urban_raw_3sec, delimiter=',',fmt="%s")
np.savetxt('urban_spect_1sec.csv', csv_urban_spect_1sec, delimiter=',',fmt="%s")
np.savetxt('urban_spect_3sec.csv', csv_urban_spect_3sec, delimiter=',',fmt="%s")

y1, sr = librosa.load(PATH_MUDI +'', duration=0.5,sr=22050*2)
y2, sr = librosa.load(PATH_MESCALINA +'', duration=0.5,sr=22050*2)
y3, sr = librosa.load(PATH_URBAN +'135160-8-0-0.wav', duration=0.5,sr=22050*2)
'''

