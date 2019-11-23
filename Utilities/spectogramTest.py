# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:41:24 2018

@author: Oblivion
"""

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
path = '../DataBase/MudiDogs/rawaudio/d23_s_19.wav'

sample_rate, samples = wavfile.read(path)
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()