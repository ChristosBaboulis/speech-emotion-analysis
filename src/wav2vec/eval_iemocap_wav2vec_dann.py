import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from src.data.iemocap_dataset_loader import load_iemocap_metadata, split_iemocap_by_sessions, CLASS_TO_IDX
from src.wav2vec.dataloaders import create_wav2vec_dataloaders
from src.wav2vec.model_wav2vec_dann import Wav2VecDANNEmotionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = r"D:\Recordings\Science\DL\IEMOCAP_full_release"
MODELS_DIR = "models"
RESULTS_DIR = "results"
MAX_AUDIO_LENGTH = 10.0

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("Using device:", DEVICE)

    # Load all samples
    samples = load_iemocap_metadata(BASE_PATH)

    # Split by session (speaker-independent, same as train_wav2vec_dann.py)
    train_samples, val_samples, test_samples = split_iemocap_by_sessions(samples)

    print("Train samples (Session1-3):", len(train_samples))
    print("Val samples (Session4):   ", len(val_samples))
    print("Test samples (Session5):   ", len(test_samples))

    # Create test DataLoader
    _, _, test_loader = create_wav2vec_dataloaders(
        train_samples, val_samples, test_samples,
        batch_size=4,
        max_audio_length=MAX_AUDIO_LENGTH
    )

    # Load best model
    best_model_path = os.path.join(MODELS_DIR, "best_wav2vec_dann_emotion.pt")
    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    print(f"Loading Wav2Vec2 DANN model from: {best_model_path}")
    model = Wav2VecDANNEmotionModel(num_classes=len(CLASS_TO_IDX), num_domains=2).to(DEVICE)
    state_dict = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    all_labels = []
    all_preds = []

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                audios, labels, attention_masks = batch
            else:
                audios, labels = batch
                attention_masks = None

            audios = audios.to(DEVICE)
            labels = labels.to(DEVICE)
            if attention_masks is not None:
                attention_masks = attention_masks.to(DEVICE)

            emotion_logits, _ = model(audios, attention_mask=attention_masks, alpha=0.0)
            preds = emotion_logits.argmax(dim=1)

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
        title="Confusion Matrix - Wav2Vec2 DANN on IEMOCAP Test Set",
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
    out_path = os.path.join(RESULTS_DIR, "confusion_matrix_wav2vec_dann_iemocap.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    main()

