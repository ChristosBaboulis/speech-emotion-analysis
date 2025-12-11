from torch.utils.data import DataLoader, WeightedRandomSampler
from src.data.iemocap_dataset import IEMOCAPDataset
import numpy as np
import torch


def create_dataloaders(train_samples, val_samples, test_samples, batch_size=16):

    # -------- TRAIN SAMPLER (BALANCING) --------
    labels = [s["label"] for s in train_samples]
    labels = np.array(labels)

    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    # -------- DATASETS --------
    train_dataset = IEMOCAPDataset(train_samples)
    val_dataset = IEMOCAPDataset(val_samples)
    test_dataset = IEMOCAPDataset(test_samples)

    # -------- LOADERS --------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
