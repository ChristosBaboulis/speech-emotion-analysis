from typing import List

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.ravdess_dataset_loader import load_ravdess_metadata
from src.data.iemocap_dataset import IEMOCAPDataset
from src.data.ravdess_dataset import RAVDESSDataset


# ---------- Create loaders ----------
def create_dann_loaders(
    train_iemocap_samples: List[dict],
    ravdess_base: str,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
):
    """
    Create DANN dataloaders for domain adaptation.
    
    Args:
        train_iemocap_samples: Pre-split IEMOCAP training samples (speaker-independent split)
        ravdess_base: Base path to RAVDESS dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        augment: Enable augmentation for source dataset (default: True for training)
    """
    # 1) RAVDESS metadata (all actors)
    ravdess_samples = load_ravdess_metadata(ravdess_base)

    # 2) Build datasets
    # domain_id: 0 = IEMOCAP, 1 = RAVDESS
    # Enable augmentation only for source dataset (training)
    source_dataset = IEMOCAPDataset(train_iemocap_samples, domain_id=0, augment=augment)
    target_dataset = RAVDESSDataset(ravdess_samples, domain_id=1, augment=False)

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
