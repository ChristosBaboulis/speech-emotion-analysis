from typing import List

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.ravdess_dataset_loader import load_ravdess_metadata
from src.wav2vec.audio_dataset import AudioDataset
from src.wav2vec.dataloaders import collate_fn


def create_wav2vec_dann_loaders(
    train_iemocap_samples: List[dict],
    ravdess_base: str,
    batch_size: int = 4,
    max_audio_length: float = 10.0,
    num_workers: int = 2,
):
    """
    Create DANN dataloaders for Wav2Vec2 domain adaptation.
    
    Args:
        train_iemocap_samples: Pre-split IEMOCAP training samples (speaker-independent split)
        ravdess_base: Base path to RAVDESS dataset
        batch_size: Batch size for dataloaders (smaller for Wav2Vec2)
        max_audio_length: Maximum audio length in seconds
        num_workers: Number of worker processes for data loading
    
    Returns:
        source_loader, target_loader
    """
    # 1) RAVDESS metadata (all actors)
    ravdess_samples = load_ravdess_metadata(ravdess_base)

    # 2) Build datasets
    # domain_id: 0 = IEMOCAP, 1 = RAVDESS
    source_dataset = AudioDataset(train_iemocap_samples, max_length=max_audio_length, domain_id=0)
    target_dataset = AudioDataset(ravdess_samples, max_length=max_audio_length, domain_id=1)

    # 3) Class balancing for source (IEMOCAP)
    labels = torch.tensor([s["label"] for s in train_iemocap_samples], dtype=torch.long)
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
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return source_loader, target_loader

