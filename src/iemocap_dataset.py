import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

class IEMOCAPDataset(Dataset):
    def __init__(self, samples, sr=16000, n_mels=128):
        self.samples = samples
        self.sr = sr
        self.n_mels = n_mels

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

        # To tensor: shape [1, n_mels, time]
        S = torch.tensor(S, dtype=torch.float32).unsqueeze(0)

        return S, label


if __name__ == "__main__":
    # Quick sanity check
    from dataset_loader import load_iemocap_metadata
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
