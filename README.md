# adaptive-noise-cancellation-python
Python-based Adaptive Noise Cancellation using the LMS algorithm. Takes a noisy voice sample and a noise-only sample to filter out background noise and output a cleaned audio signal. Simple, effective, and easy to run.



# Install dependencies
!apt-get install ffmpeg -y > /dev/null
!pip install numpy scipy matplotlib soundfile pydub --quiet

from google.colab import files
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os

# 📁 Upload .m4a or .wav files
print("Upload your noisy speech file (.m4a or .wav):")
uploaded_noisy = files.upload()

print("Upload your noise-only reference file (.m4a or .wav):")
uploaded_noise = files.upload()

# Convert to .wav if needed
def convert_to_wav(filepath):
    if filepath.endswith(".wav"):
        return filepath
    audio = AudioSegment.from_file(filepath)
    wav_path = filepath.rsplit('.', 1)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

noisy_path = convert_to_wav(list(uploaded_noisy.keys())[0])
noise_path = convert_to_wav(list(uploaded_noise.keys())[0])

#  Load files
desired, fs1 = sf.read(noisy_path)
reference, fs2 = sf.read(noise_path)

assert fs1 == fs2, "Sampling rates must match"

# Convert to mono if stereo
if len(desired.shape) > 1:
    desired = desired[:, 0]
if len(reference.shape) > 1:
    reference = reference[:, 0]

#  Trim to same length
min_len = min(len(desired), len(reference))
desired = desired[:min_len]
reference = reference[:min_len]

#  LMS Filter
def lms_filter(desired, reference, mu=0.0001, filter_order=64):
    n = len(reference)
    y = np.zeros(n)
    e = np.zeros(n)
    w = np.zeros(filter_order)

    for i in range(filter_order, n):
        x = reference[i-filter_order:i][::-1]
        y[i] = np.dot(w, x)
        e[i] = desired[i] - y[i]
        w = w + 2 * mu * e[i] * x
    return e  # Cleaned signal

#  Apply filtering
cleaned = lms_filter(desired, reference)

#  Save cleaned audio
output_filename = "cleaned_output.wav"
sf.write(output_filename, cleaned, fs1)

#  Download
files.download(output_filename)

#  Plot
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.title("Original Noisy Speech")
plt.plot(desired)
plt.subplot(3, 1, 2)
plt.title("Reference Noise")
plt.plot(reference)
plt.subplot(3, 1, 3)
plt.title("Cleaned Output")
plt.plot(cleaned)
plt.tight_layout()
plt.show()
