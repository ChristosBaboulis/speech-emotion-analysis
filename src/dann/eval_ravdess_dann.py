import os
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import warnings

from data.dataset_ravdess import load_ravdess_metadata
from dann.model_dann import DANNEmotionModel
from utils.evaluate import eval_with_confusion   # <= USE confusion

warnings.filterwarnings(
    "ignore",
    message="At least one mel filterbank has all zero values.*"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAVDESS_BASE = r"D:\Recordings\Science\DL\RAVDESS"


def load_wav_to_logmel(path: str,
                       sample_rate: int = 16000,
                       n_mels: int = 128,
                       max_frames: int = 300) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)   # [C, T]

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=n_mels,
    )(waveform)

    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

    t = mel_spec.size(2)
    if t < max_frames:
        pad = max_frames - t
        mel_spec = torch.nn.functional.pad(mel_spec, (0, pad))
    else:
        mel_spec = mel_spec[:, :, :max_frames]

    return mel_spec  # [1, 128, max_frames]


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


class DANNEmotionWrapper(nn.Module):
    """
    Wrapper over DANN for inference:
    forward(x) -> *only* emotion logits
    """
    def __init__(self, dann_model: DANNEmotionModel):
        super().__init__()
        self.dann_model = dann_model

    def forward(self, x):
        emotion_logits, _ = self.dann_model(x, alpha=0.0)
        return emotion_logits


def main():
    print("Device:", DEVICE)

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

    # Load DANN
    dann = DANNEmotionModel(num_classes=5, num_domains=2).to(DEVICE)
    ckpt_path = os.path.join("models", "best_dann_emotion_cnn.pt")
    print("Loading DANN model from:", ckpt_path)
    dann.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    # Wrapper for using with eval_with_confusion
    model = DANNEmotionWrapper(dann).to(DEVICE)

    CLASS_NAMES = ["angry", "happy", "neutral", "sad", "frustrated"]

    acc, cm = eval_with_confusion(
        model=model,
        loader=loader,
        device=DEVICE,
        num_classes=5,
        class_names=CLASS_NAMES,
        normalize=True,
        title="DANN CNN on RAVDESS",
    )

    print(f"DANN model accuracy on RAVDESS: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()
