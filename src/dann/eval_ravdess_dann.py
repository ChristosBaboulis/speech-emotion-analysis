import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings

from src.data.ravdess_dataset_loader import load_ravdess_metadata
from src.data.ravdess_dataset import RAVDESSDataset
from src.dann.model_dann import DANNEmotionModel
from src.baseline.evaluate import eval_with_confusion

warnings.filterwarnings(
    "ignore",
    message="At least one mel filterbank has all zero values.*"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAVDESS_BASE = r"D:\Recordings\Science\DL\RAVDESS"


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

    # Use the same dataset class as training for consistent preprocessing
    dataset = RAVDESSDataset(samples)
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
