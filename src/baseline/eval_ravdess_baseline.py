import os
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings

from src.data.ravdess_dataset_loader import load_ravdess_metadata
from src.data.ravdess_dataset import RAVDESSDataset
from src.baseline.model_crnn import EmotionCRNN
from src.baseline.evaluate import eval_with_confusion

warnings.filterwarnings(
    "ignore",
    message="At least one mel filterbank has all zero values.*"
)

# ----------------- PATHS & DEVICE -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAVDESS_BASE = r"D:\Recordings\Science\DL\RAVDESS"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("Device:", DEVICE)

    # 1) load metadata RAVDESS
    samples = load_ravdess_metadata(RAVDESS_BASE)
    print(f"Total RAVDESS samples: {len(samples)}")

    # Use the same dataset class as training for consistent preprocessing
    dataset = RAVDESSDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2) load baseline CNN
    model = EmotionCRNN(num_classes=5).to(DEVICE)
    ckpt_path = os.path.join("models", "best_emotion_cnn.pt")
    print("Loading baseline CNN from:", ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    # 3) confusion matrix + accuracy
    CLASS_NAMES = ["angry", "happy", "neutral", "sad", "frustrated"]
    save_path = os.path.join(RESULTS_DIR, "confusion_matrix_baseline_ravdess.png")

    acc, cm = eval_with_confusion(
        model=model,
        loader=loader,
        device=DEVICE,
        num_classes=5,
        class_names=CLASS_NAMES,
        normalize=False,
        title="Baseline CNN on RAVDESS",
        save_path=save_path,
    )

    print(f"Baseline CNN accuracy on RAVDESS: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()
