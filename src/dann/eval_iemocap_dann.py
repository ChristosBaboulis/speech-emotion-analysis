import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from src.data.iemocap_dataset_loader import load_iemocap_metadata, split_iemocap_by_sessions, CLASS_TO_IDX
from src.baseline.dataloaders import create_dataloaders
from src.dann.model_dann import DANNEmotionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = r"D:\Recordings\Science\DL\IEMOCAP_full_release"
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


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
    print("Using device:", DEVICE)

    # Load all samples
    samples = load_iemocap_metadata(BASE_PATH)

    # Split by session (speaker-independent, same as train_dann.py)
    train_samples, val_samples, test_samples = split_iemocap_by_sessions(samples)

    print("Train samples (Session1-3):", len(train_samples))
    print("Val samples (Session4):   ", len(val_samples))
    print("Test samples (Session5):   ", len(test_samples))

    # Create only the test DataLoader (train/val are unused here)
    _, _, test_loader = create_dataloaders(
        train_samples, val_samples, test_samples, batch_size=32
    )

    # Load best DANN model
    best_model_path = os.path.join(MODELS_DIR, "best_dann_emotion_cnn.pt")
    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(f"Best DANN model not found at {best_model_path}")

    dann = DANNEmotionModel(num_classes=len(CLASS_TO_IDX), num_domains=2).to(DEVICE)
    state_dict = torch.load(best_model_path, map_location=DEVICE)
    dann.load_state_dict(state_dict)
    
    # Wrapper for consistent interface
    model = DANNEmotionWrapper(dann).to(DEVICE)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for mel, labels in test_loader:
            mel = mel.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(mel)
            preds = outputs.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    # Overall test accuracy
    test_acc = (all_labels == all_preds).mean()
    print(f"Test accuracy: {test_acc:.4f}")

    # Class names from CLASS_TO_IDX
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Classification report
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, target_names)


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix - DANN on IEMOCAP Test Set",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Print numbers inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "confusion_matrix_dann_iemocap.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    main()

