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

path_mescalina2017 = 'C:\\Users\\Templar\\Desktop\\mescalinaClasificados2017Corregida_Nov2017\\'
files_mescalina2017 = []
mescalina2017_individuos = []
mescalina2017_contexto = []
sec1_mescalina2017 = []
for item in os.listdir(path_mescalina2017+'/'):
    for subitem in os.listdir(path_mescalina2017+'/'+item):       
        for subsubitem in os.listdir(path_mescalina2017+'/'+item+'/'+subitem):
            if 'L-S' == subitem:
                continue
            elif 'L-W' == subitem:
                continue
            elif 'L' in subitem:
                mescalina2017_individuos.append(item)
                mescalina2017_contexto.append(subitem)               
                files_mescalina2017.append(subsubitem)
                print(len(files_mescalina2017))
                sec1, sr = librosa.load(path_mescalina2017+'/'+item+'/'+subitem+'/'+subsubitem,sr=8820,duration=1.0)
                emptyspace = np.zeros(int(8820 - len(sec1)))
                sec1 = np.concatenate((sec1, emptyspace), axis=0)
                sec1 = sec1/abs(sec1).max()
                sec1_mescalina2017.append(sec1)
                scaled = np.int16(sec1/np.max(np.abs(sec1)) * 32767)
                #write('../DataBase/mescalina2017/'+item+'_'+subitem+'_'+subsubitem, 8820, scaled)

                
mescalina2017_individuos_num = pd.factorize(mescalina2017_individuos)
mescalina2017_sex = []
for name in mescalina2017_individuos:
    name = np.array2string(name)
    if ('LAZZY' in name or 'LUNA' in name or 'MEGARA' in name or 'CHUFINA' in name or 'GUERA' in name or 'MALY' in name or 
        'KIARA' in name or 'DAYSI' in name or 'SOFIA' in name or 'MORINGA' in name or 'MIKA' in name or 'BELLA' in name or 
        'PHIBI' in name or 'NENE' in name or 'MAMAFISGON' in name or 'CHIKI' in name or 'TINKY' in name or 'CHICA' in name or 
        'PALOMA' in name or 'GUERA2' in name or 'PRINCESA' in name or 'TAINY' in name or 'PANTERA' in name or 'ESTRELLA' in name or 
        'PALOMA2' in name or 'KIKA' in name or 'CAMPANA' in name or 'ROBINA' in name or 'MISHA' in name or 'LILY' in name or 
        'PAGE' in name or 'BECKY' in name or 'JADE' in name or 'MORA' in name or 'PELUSA' in name or 'NANA' in name or 
        'EVEREST' in name or 'BEILY' in name or 'COSI' in name or 'PELUDINA' in name or 'CHABELA' in name or 'KYLEY' in name or 
        'MUNE' in name or 'GALA' in name or 'ZARA' in name or 'RUTILA' in name or 'CLOE' in name or 'PIRRUNA' in name or 
        'PEQUE' in name or 'CHIQUITA' in name or 'LASY' in name):
        mescalina2017_sex.append('F')
    elif ('KOSTER' in name or 'JERRY' in name or 'BADY' in name or 'KIZZY' in name or 'PERRY' in name or 'CAPY' in name or 
        'HOMERO' in name or 'KLEIN' in name or 'BENITO' in name or 'TURUGUS' in name or 'FISGON' in name or 'KIEN' in name or 
        'GOOFY' in name or 'KIKO' in name or 'SANZON' in name or 'CHOCOLATE' in name or 'COFFE' in name or 'TAIZON' in name or 
        'GRENAS' in name or 'MICKEY' in name or 'PELUDIN' in name or 'CENTAVO' in name or 'CACHITO' in name or 'ZEBRY' in name or 
        'MAIKY' in name or 'NINO' in name or 'LUCAS' in name or 'TITO' in name or 'RAYO' in name or 'CENTAVITO' in name):
        mescalina2017_sex.append('M')
        
mescalina2017_sex = (np.asarray(mescalina2017_sex)).astype(str)

print(mescalina2017_individuos_num[1])
mescalina2017_contexto_num = pd.factorize(mescalina2017_contexto)
print(mescalina2017_contexto_num[1])    
mescalina2017_individuos = np.reshape(np.asarray(mescalina2017_individuos),(len(mescalina2017_individuos),1))
mescalina2017_individuos_num = np.reshape(np.asarray(mescalina2017_individuos_num[0]),(len(mescalina2017_individuos_num[0]),1))
mescalina2017_contexto = np.reshape(np.asarray(mescalina2017_contexto),(len(mescalina2017_contexto),1))
mescalina2017_contexto_num = np.reshape(np.asarray(mescalina2017_contexto_num[0]),(len(mescalina2017_contexto_num[0]),1))
mescalina2017_sex = np.reshape(np.asarray(mescalina2017_contexto),(len(mescalina2017_contexto),1))
mescalina2017_sex_num = np.reshape(np.asarray(mescalina2017_contexto_num[0]),(len(mescalina2017_contexto_num[0]),1))
files_mescalina2017 = np.reshape(np.asarray(files_mescalina2017),(len(files_mescalina2017),1))
csv_mescalina2017 = np.concatenate((files_mescalina2017,
                                mescalina2017_individuos,mescalina2017_individuos_num,
                                mescalina2017_contexto,mescalina2017_contexto_num,
                                mescalina2017_sex,mescalina2017_sex_num), axis=1)
sec1_mescalina2017 = np.reshape(sec1_mescalina2017,(len(files_mescalina2017),len(sec1)))
#csv_mescalina2017 = np.reshape(np.asarray(files_mescalina2017),(len(files_mescalina2017),1))
np.savetxt('mescalina2017.csv', csv_mescalina2017, delimiter=',',fmt="%s")
mescalina2017 = genfromtxt('./mescalina2017.csv', delimiter=',',dtype='str')    
context_categories = csv_mescalina2017[:,4]
individ_categories = csv_mescalina2017[:,2]
mescalina2017context = np.column_stack((sec1_mescalina2017,context_categories))
np.savetxt('mescalina2017_context_raw.csv', mescalina2017context, delimiter=',',fmt="%s")
del mescalina2017context
print('generating individ file')
mescalina2017individ = np.column_stack((sec1_mescalina2017,individ_categories))
np.savetxt('mescalina2017_individ_raw.csv', mescalina2017individ, delimiter=',',fmt="%s")
del mescalina2017individ

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
y2, sr = librosa.load(PATH_mescalina2017 +'', duration=0.5,sr=22050*2)
y3, sr = librosa.load(PATH_URBAN +'135160-8-0-0.wav', duration=0.5,sr=22050*2)
'''

