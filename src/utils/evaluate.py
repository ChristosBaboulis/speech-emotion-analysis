import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from data.dataset_loader import load_iemocap_metadata, CLASS_TO_IDX
from baseline.dataloaders import create_dataloaders
from baseline.model_cnn import EmotionCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = r"D:\Recordings\Science\DL\IEMOCAP_full_release"
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("Using device:", DEVICE)

    # Load all samples and reproduce the same 70/15/15 split (same random_state as train.py)
    samples = load_iemocap_metadata(BASE_PATH)

    train_samples, temp = train_test_split(
        samples, test_size=0.30, shuffle=True, random_state=42
    )
    val_samples, test_samples = train_test_split(
        temp, test_size=0.50, shuffle=True, random_state=42
    )

    print("Test samples:", len(test_samples))

    # Create only the test DataLoader (train/val are unused here)
    _, _, test_loader = create_dataloaders(
        train_samples, val_samples, test_samples, batch_size=32
    )

    # Load best model
    best_model_path = os.path.join(MODELS_DIR, "best_emotion_cnn.pt")
    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    model = EmotionCNN(num_classes=len(CLASS_TO_IDX)).to(DEVICE)
    state_dict = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for mel, labels in test_loader:
            mel = mel.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(mel)
            _, preds = outputs.max(1)

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
        title="Confusion Matrix (Test Set)",
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
    out_path = os.path.join(RESULTS_DIR, "confusion_matrix_test.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")


def eval_with_confusion(model, loader, device, num_classes, class_names=None,
                        normalize=False, title="Confusion matrix"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for mel, labels in loader:           # mel: [B, 1, 128, 300]
            mel = mel.to(device)
            labels = labels.to(device)

            outputs = model(mel)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = (all_preds == all_labels).mean()

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(num_classes))
    ).astype(float)

    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    # ---- plot ----
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    plt.show()

    return acc, cm

if __name__ == "__main__":
    main()
