# Audio Analysis And Noise Reduction

## AIM

To perform **audio noise reduction and analysis** using Python by applying noise reduction techniques and visualizing the results in both time and frequency domains.

---
## TASK

- Upload and load an audio signal  
- Apply noise reduction and optional gain processing  
- Save and compare original and processed audio  
- Perform STFT to analyze frequency components  
- Plot spectrograms of original, processed, and noise signals  
- Visualize signals in time domain and compare results  

---

## APPARATUS REQUIRED

- PC with Python (Google Colab / Jupyter Notebook)  
- Libraries:
  - `librosa`
  - `noisereduce`
  - `pedalboard`
  - `soundfile`
  - `matplotlib`
  - `numpy`

---

## THEORY

Noise in audio signals reduces clarity and quality. Noise reduction techniques remove unwanted components while preserving useful information.

- **Noise Reduction:** Removes background noise  
- **STFT:** Converts signal to time-frequency domain  
- **Spectrogram:** Shows frequency variation over time  
- **dB Scale:** Helps visualize signal intensity  
- **Gain:** Adjusts signal amplitude  

---

## PROGRAM: 

~~~
!pip install noisereduce pedalboard librosa soundfile

import librosa
import librosa.display
import noisereduce as nr
from pedalboard import Pedalboard, Gain
from IPython.display import Audio, display
from google.colab import files
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

print("Please upload your audio file (.wav recommended)")
uploaded = files.upload()

if uploaded:
    file_name = list(uploaded.keys())[0]
    file_path = file_name
    print(f"Loading audio from: {file_path}")
else:
    raise FileNotFoundError("No audio file uploaded.")

y, sr = librosa.load(file_path, sr=None)

y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=1.0)

board = Pedalboard([Gain(gain_db=0)])  # Change gain if needed
y_processed = board(y_denoised, sr)

output_file_name = "output_denoised_audio.wav"
sf.write(output_file_name, y_processed, sr)

print("Original Audio:")
display(Audio(file_path))

print("Denoised Audio:")
display(Audio(output_file_name))

D_original = librosa.stft(y)
D_processed = librosa.stft(y_processed)

DB_original = librosa.amplitude_to_db(np.abs(D_original), ref=np.max)
DB_processed = librosa.amplitude_to_db(np.abs(D_processed), ref=np.max)

plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
librosa.display.specshow(DB_original, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Original Audio Spectrogram')

plt.subplot(2, 1, 2)
librosa.display.specshow(DB_processed, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Denoised Audio Spectrogram')

plt.tight_layout()
plt.show()


noise_signal = y - y_processed
D_noise = librosa.stft(noise_signal)
DB_noise = librosa.amplitude_to_db(np.abs(D_noise), ref=np.max)

plt.figure(figsize=(12, 6))
librosa.display.specshow(DB_noise, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Estimated Noise Spectrogram')
plt.tight_layout()
plt.show()

time = np.linspace(0, len(y) / sr, num=len(y))

plt.figure(figsize=(15, 5))

plt.subplot(2, 1, 1)
plt.plot(time, y, alpha=0.7)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, y_processed, alpha=0.7)
plt.title('Denoised Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()

print("Noise reduction completed successfully!")
~~~

# OUTPUT: 

<img width="1351" height="790" alt="download" src="https://github.com/user-attachments/assets/a73ddddd-b4e1-4fe5-8d69-40a23a1f20b4" />
<img width="1105" height="590" alt="download" src="https://github.com/user-attachments/assets/30a20158-5d63-4e33-af4e-1ba0cc074183" />
<img width="1489" height="490" alt="download" src="https://github.com/user-attachments/assets/23380b15-82c9-491d-b945-f53e7d260d3e" />

---

## FUTURE IMPROVEMENTS

- Real-time noise reduction  
- Advanced filtering techniques  
- GUI-based implementation

---
  
# RESULT: 
  Thus Audio noise reduction was effectively implemented, and analysis confirms improved signal quality in both time and frequency domains.
