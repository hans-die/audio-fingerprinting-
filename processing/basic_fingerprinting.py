# Basic Fingerprinting Research
import librosa
import numpy as np
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt

# Extraigo el audio y el sample rate
y, sr = librosa.load("/Users/hansdietrich/Documents/GitHub/audio-fingerprinting-/songs/Twin Peaks Intro.mp3")


# Grafico
#plt.plot(y)
#plt.title("Waveform")
#plt.show()

# Necesitamos mas info -> Usamos fourier
S = librosa.stft(y)
S_db = librosa.amplitude_to_db(abs(S))  # La STFT es compleja por lo que lo tengo que pasar a valor absoluto

# librosa.display.specshow(S_db, sr = sr, y_axis = 'log', x_axis = 'time' )
# plt.colorbar()
# plt.title('Spectrogram')
# plt.show()

# Nos vamos a quedar solo con lo picos, ya que son mas resistentes al ruido
umbral = -5
neighbourhood_size = 40
local_max = maximum_filter(S_db, size = neighbourhood_size) == S_db
detected_peaks = local_max & (S_db > umbral)
freq_idx, time_idx = np.where(detected_peaks)

plt.figure(figsize = (10,6))
plt.imshow(S_db, aspect = 'auto', origin = 'lower')
plt.scatter(time_idx, freq_idx, s = 5, c = 'red')
plt.title('Peaks on a Spectrogram')
plt.show()

