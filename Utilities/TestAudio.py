import librosa
import librosa.display
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

mel = 128
path = '../DataBase/mudidogs/rawaudio'
file = 'C:\\Users\\Oblivion\\Documents\\GitHub\\DataBase\\mudidogs\\context\\play\\d20\\saba_play_031007.wav'

audio44khz, sr44 = librosa.load(file,sr=44100,duration=1,offset=1.2)
audio8khz, sr8 = librosa.load(file,sr=8820,duration=1,offset=1.2)
'''
plt.subplot(221)
plt.title('Muestra Original a 44100Hz')
plt.plot(audio44khz,linewidth=0.1)

plt.subplot(222)
plt.title('44100Hz')
plt.specgram(audio8khz,Fs=44100)

plt.subplot(222)
melspec44khz = librosa.feature.melspectrogram(y=audio44khz, sr=sr44, n_mels=128,fmax=sr44)
#plt.imshow(melspec44khz,aspect= 'auto')
librosa.display.specshow(librosa.power_to_db(melspec44khz,ref=np.max),y_axis='mel', fmax=sr44)
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma de Mel a 44100Hz')
plt.tight_layout()
'''
plt.subplot(311)
plt.title('Muestra Reducida a 8820Hz')
plt.plot(audio8khz,linewidth=0.1)

plt.subplot(312)
melspec8khz = librosa.feature.melspectrogram(y=audio8khz, sr=sr8, n_mels=128,fmax=sr8)
plt.title('Espectrograma en escala de Mel')
#plt.specgram(audio8khz,Fs=8820)
plt.imshow(melspec8khz,aspect='auto',cmap='jet')

plt.subplot(313)
#plt.imshow(melspec8khz,aspect= 'auto')
librosa.display.specshow(librosa.power_to_db(melspec8khz,ref=np.max),y_axis='mel', fmax=sr8)
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma en dB a 8820Hz')
plt.tight_layout()