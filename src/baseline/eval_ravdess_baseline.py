import os
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import warnings

from src.data.ravdess_dataset_loader import load_ravdess_metadata
from baseline.model_cnn import EmotionCNN
from utils.evaluate import eval_with_confusion   # <= USE confusion

warnings.filterwarnings(
    "ignore",
    message="At least one mel filterbank has all zero values.*"
)

# ----------------- PATHS & DEVICE -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAVDESS_BASE = r"D:\Recordings\Science\DL\RAVDESS"


# ----------------- AUDIO → LOG-MEL -----------------
def load_wav_to_logmel(path: str,
                       sample_rate: int = 16000,
                       n_mels: int = 128,
                       max_frames: int = 300) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)   # [C, T]

    # stereo -> mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    # mel spectrogram
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=n_mels,
    )(waveform)

    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

    # pad / truncate
    t = mel_spec.size(2)
    if t < max_frames:
        pad = max_frames - t
        mel_spec = torch.nn.functional.pad(mel_spec, (0, pad))
    else:
        mel_spec = mel_spec[:, :, :max_frames]

    return mel_spec  # [1, 128, max_frames]


# ----------------- DATASET -----------------
class RAVDESSMelDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        wav_path = item["path"]
        label = int(item["label"])

        mel = load_wav_to_logmel(wav_path)
        return mel, label


def main():
    print("Device:", DEVICE)

    # 1) load metadata RAVDESS
    samples = load_ravdess_metadata(RAVDESS_BASE)
    print(f"Total RAVDESS samples: {len(samples)}")

    dataset = RAVDESSMelDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2) load baseline CNN
    model = EmotionCNN(num_classes=5).to(DEVICE)
    ckpt_path = os.path.join("models", "best_emotion_cnn.pt")
    print("Loading baseline CNN from:", ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    # 3) confusion matrix + accuracy
    CLASS_NAMES = ["angry", "happy", "neutral", "sad", "frustrated"]

    acc, cm = eval_with_confusion(
        model=model,
        loader=loader,
        device=DEVICE,
        num_classes=5,
        class_names=CLASS_NAMES,
        normalize=True,
        title="Baseline CNN on RAVDESS",
    )

    print(f"Baseline CNN accuracy on RAVDESS: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()
