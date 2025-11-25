import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Path to audio file
AUDIO_PATH = os.path.join("data", "example.wav")

# Load audio
y, sr = librosa.load(AUDIO_PATH, sr=16000)

# Compute Mel-spectrogram
S = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=1024,
    hop_length=256,
    n_mels=128
)

# Convert to log scale
S_db = librosa.power_to_db(S, ref=np.max)

# Plot and save
plt.figure(figsize=(8, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel-Spectrogram")
plt.tight_layout()

# Save to results folder
os.makedirs("results", exist_ok=True)
plt.savefig("results/example_melspec.png")

print("Saved mel-spectrogram to results/example_melspec.png")
