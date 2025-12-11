import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import librosa
import numpy as np


class IEMOCAPDataset(Dataset):
    def __init__(self, samples, sr=16000, n_mels=128, max_frames=300):
        self.samples = samples
        self.sr = sr
        self.n_mels = n_mels
        self.max_frames = max_frames  # fixed time dimension

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        wav_path = item["path"]
        label = item["label"]

        # Load audio
        y, sr = librosa.load(wav_path, sr=self.sr)

        # Mel-spectrogram
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=self.n_mels,
        )

        # Log-Mel
        S = librosa.power_to_db(S, ref=np.max)

        # To tensor: [n_mels, time]
        S = torch.tensor(S, dtype=torch.float32)

        # Pad or cut to fixed number of frames
        T = S.shape[1]
        if T < self.max_frames:
            pad = self.max_frames - T
            # pad last dimension (time)
            S = F.pad(S, (0, pad))
        else:
            S = S[:, : self.max_frames]

        # Final shape: [1, n_mels, max_frames]
        S = S.unsqueeze(0)

        return S, label


if __name__ == "__main__":
    from data.dataset_loader import load_iemocap_metadata
    from sklearn.model_selection import train_test_split

    BASE_PATH = r"D:\Recordings\Science\DL\IEMOCAP_full_release"

    samples = load_iemocap_metadata(BASE_PATH)
    train_samples, temp = train_test_split(
        samples, test_size=0.30, shuffle=True, random_state=42
    )
    val_samples, test_samples = train_test_split(
        temp, test_size=0.50, shuffle=True, random_state=42
    )

    dataset = IEMOCAPDataset(train_samples)
    mel, label = dataset[0]

    print("Mel shape:", mel.shape)
    print("Label:", label)
