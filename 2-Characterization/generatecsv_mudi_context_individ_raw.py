import pandas as pd
import librosa
import os
import numpy as np
from numpy import genfromtxt
from scipy.io.wavfile import write
from pydub import AudioSegment

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
    sec1 = sec1/abs(sec1).max()
    sec1_mudi.append(sec1)
    scaled = np.int16(sec1/np.max(np.abs(sec1)) * 32767)
    write(path_mudi+'/1sec/'+item, 8820, scaled)

sec1_mudi = np.reshape(sec1_mudi,(len(files_mudi),len(sec1)))
csv_mudi = np.reshape(np.asarray(files_mudi),(len(files_mudi),1))
mudi = genfromtxt('./mudidogs.csv', delimiter=',',dtype='str')    
context_categories = mudi[:,2]
individ_categories = mudi[:,4]
print('generating context file')
mudicontext = np.column_stack((sec1_mudi,context_categories))
np.savetxt('mudi_context_raw.csv', mudicontext, delimiter=',',fmt="%s")
del mudicontext
print('generating individ file')
mudiindivid = np.column_stack((sec1_mudi,individ_categories))
np.savetxt('mudi_individ_raw.csv', mudiindivid, delimiter=',',fmt="%s")
