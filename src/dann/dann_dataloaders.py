import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="At least one mel filterbank has all zero values.*",
)


from src.data.dataset_loader import load_iemocap_metadata
from src.data.dataset_ravdess import load_ravdess_metadata


# ---------- Audio → log-mel ----------
def load_wav_to_logmel(path: str,
                       sample_rate: int = 16000,
                       n_mels: int = 128,
                       max_frames: int = 300) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    # ensure mono: average channels if stereo
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=n_mels,
    )(waveform)

    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

    # mel_spec shape: [1, n_mels, time]
    t = mel_spec.size(2)
    if t < max_frames:
        pad = max_frames - t
        mel_spec = torch.nn.functional.pad(mel_spec, (0, pad))
    else:
        mel_spec = mel_spec[:, :, :max_frames]

    return mel_spec  # [1, 128, max_frames]


# ---------- Dataset ----------
class EmotionDomainDataset(Dataset):
    """
    Generic dataset for DANN:
    returns (log_mel, emotion_label, domain_label)
    domain_label: 0 = IEMOCAP, 1 = RAVDESS
    """

    def __init__(self, samples: List[dict], domain_id: int):
        self.samples = samples
        self.domain_id = domain_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        item = self.samples[idx]
        wav_path = item["path"]
        label = int(item["label"])

        mel = load_wav_to_logmel(wav_path)

        return mel, label, self.domain_id


# ---------- Helpers to filter IEMOCAP by session ----------
def filter_iemocap_by_sessions(samples: List[dict], keep_sessions: List[str]) -> List[dict]:
    keep_set = set(keep_sessions)
    filtered = []
    for s in samples:
        path = s["path"]
        # path contains .../SessionX/...
        for sess in keep_set:
            if os.path.sep + sess + os.path.sep in path:
                filtered.append(s)
                break
    return filtered


# ---------- Create loaders ----------
def create_dann_loaders(
    iemocap_base: str,
    ravdess_base: str,
    batch_size: int = 32,
    num_workers: int = 4,
):
    # 1) Load IEMOCAP metadata and split by session
    all_iemocap = load_iemocap_metadata(iemocap_base)

    train_iemocap = filter_iemocap_by_sessions(
        all_iemocap, ["Session1", "Session2", "Session3"]
    )

    # 2) RAVDESS metadata (all actors)
    ravdess_samples = load_ravdess_metadata(ravdess_base)

    # 3) Build datasets
    source_dataset = EmotionDomainDataset(train_iemocap, domain_id=0)
    target_dataset = EmotionDomainDataset(ravdess_samples, domain_id=1)

    # 4) Class balancing for source (IEMOCAP)
    labels = torch.tensor([s["label"] for s in train_iemocap], dtype=torch.long)
    num_classes = int(labels.max().item()) + 1
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return source_loader, target_loader
