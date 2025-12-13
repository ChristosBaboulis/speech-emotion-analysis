import os

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from src.data.ravdess_dataset_loader import load_ravdess_metadata
from src.wav2vec.audio_dataset import AudioDataset
from src.wav2vec.model_wav2vec_dann import Wav2VecDANNEmotionModel
from src.wav2vec.dataloaders import collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAVDESS_BASE = r"D:\Recordings\Science\DL\RAVDESS"
RESULTS_DIR = "results"
MAX_AUDIO_LENGTH = 10.0

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("Device:", DEVICE)

    # Load RAVDESS metadata
    samples = load_ravdess_metadata(RAVDESS_BASE)
    print(f"Total RAVDESS samples: {len(samples)}")

    # Use AudioDataset for raw audio loading (same as training)
    dataset = AudioDataset(samples, max_length=MAX_AUDIO_LENGTH)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Load Wav2Vec2 DANN model
    model = Wav2VecDANNEmotionModel(num_classes=5, num_domains=2).to(DEVICE)
    ckpt_path = os.path.join("models", "best_wav2vec_dann_emotion.pt")
    print(f"Loading Wav2Vec2 DANN model from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    CLASS_NAMES = ["angry", "happy", "neutral", "sad", "frustrated"]
    
    # Evaluation loop
    all_preds = []
    all_labels = []

    print("\nEvaluating on RAVDESS test set...")
    with torch.no_grad():
        for batch in loader:
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

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = (all_preds == all_labels).mean()
    print(f"Wav2Vec2 DANN accuracy on RAVDESS: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Wav2Vec2 DANN on RAVDESS",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = "d"
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
    save_path = os.path.join(RESULTS_DIR, "confusion_matrix_wav2vec_dann_ravdess.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()

