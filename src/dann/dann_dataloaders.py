import os
from typing import List

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.iemocap_dataset_loader import load_iemocap_metadata
from src.data.ravdess_dataset_loader import load_ravdess_metadata
from src.data.iemocap_dataset import IEMOCAPDataset
from src.data.ravdess_dataset import RAVDESSDataset


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
    # domain_id: 0 = IEMOCAP, 1 = RAVDESS
    source_dataset = IEMOCAPDataset(train_iemocap, domain_id=0)
    target_dataset = RAVDESSDataset(ravdess_samples, domain_id=1)

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
