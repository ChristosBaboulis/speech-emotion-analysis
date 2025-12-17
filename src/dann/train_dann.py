import os
import torch
import torch.nn as nn
from torch.optim import Adam
import math

# Fix OpenMP warning on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Use absolute imports so the src package resolves correctly.
from src.data.iemocap_dataset_loader import load_iemocap_metadata, split_iemocap_by_sessions
from src.baseline.dataloaders import create_dataloaders
from src.dann.dann_dataloaders import create_dann_loaders
from src.dann.model_dann import DANNEmotionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IEMOCAP_BASE = r"D:\Recordings\Science\DL\IEMOCAP_full_release"
RAVDESS_BASE = r"D:\Recordings\Science\DL\RAVDESS"

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_dann_emotion_cnn.pt")

NUM_EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-3
LAMBDA_DOMAIN = 0.5


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for mel, labels in loader:
            mel = mel.to(DEVICE)
            labels = labels.to(DEVICE)

            emotion_logits, _ = model(mel, alpha=0.0)
            loss = criterion(emotion_logits, labels)

            total_loss += loss.item() * mel.size(0)
            preds = emotion_logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_acc


def main():
    print("Using device:", DEVICE)

    # ---------- Load IEMOCAP and make speaker-independent split ----------
    samples = load_iemocap_metadata(IEMOCAP_BASE)
    train_samples, val_samples, test_samples = split_iemocap_by_sessions(samples)

    print("Train samples:", len(train_samples))
    print("Val samples:  ", len(val_samples))
    print("Test samples: ", len(test_samples))

    # Dataloaders for evaluation (source domain only)
    _, val_loader_src, test_loader_src = create_dataloaders(
        train_samples, val_samples, test_samples, batch_size=BATCH_SIZE
    )

    # ---------- DANN loaders (source: IEMOCAP train, target: RAVDESS) ----------
    # num_workers=0 for Windows compatibility (avoids OpenMP conflicts)
    src_loader, tgt_loader = create_dann_loaders(
        train_samples,  # Pre-split IEMOCAP training samples (Session1-3)
        RAVDESS_BASE,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print("Source DANN batches:", len(src_loader))
    print("Target DANN batches:", len(tgt_loader))

    # ---------- Model, losses, optimizer ----------
    model = DANNEmotionModel(num_classes=5, num_domains=2).to(DEVICE)
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_cls_loss = 0.0
        total_dom_loss = 0.0
        total_correct = 0
        total_samples = 0

        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)

        num_steps = len(src_loader)

        for step in range(num_steps):
            try:
                src_mel, src_labels, src_domains = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                src_mel, src_labels, src_domains = next(src_iter)

            try:
                tgt_mel, tgt_labels, tgt_domains = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_mel, tgt_labels, tgt_domains = next(tgt_iter)

            src_mel = src_mel.to(DEVICE)
            src_labels = src_labels.to(DEVICE)
            src_domains = src_domains.to(DEVICE)

            tgt_mel = tgt_mel.to(DEVICE)
            tgt_domains = tgt_domains.to(DEVICE)

            # Concatenate for domain classification
            all_mel = torch.cat([src_mel, tgt_mel], dim=0)
            all_domains = torch.cat([src_domains, tgt_domains], dim=0)

            # Alpha scheduling for gradient reversal (standard DANN approach)
            # Progress from 0 to 1 across training: p in [0, 1]
            p = (epoch - 1 + step / num_steps) / NUM_EPOCHS
            # Sigmoid-like schedule: alpha increases smoothly from 0 to MAX_ALPHA
            alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0  # [0, ~1] for p in [0, 1]
            MAX_ALPHA = 0.7  # Cap alpha to prevent too strong domain alignment early
            alpha = min(alpha, MAX_ALPHA)

            optimizer.zero_grad()

            emotion_logits, domain_logits = model(all_mel, alpha=alpha)

            # classification loss only on source samples
            bs_src = src_mel.size(0)
            cls_logits = emotion_logits[:bs_src]

            loss_cls = class_criterion(cls_logits, src_labels)
            loss_dom = domain_criterion(domain_logits, all_domains)

            loss = loss_cls + LAMBDA_DOMAIN * loss_dom
            loss.backward()
            optimizer.step()

            # stats
            total_cls_loss += loss_cls.item()
            total_dom_loss += loss_dom.item()

            preds = cls_logits.argmax(dim=1)
            total_correct += (preds == src_labels).sum().item()
            total_samples += bs_src

        avg_cls_loss = total_cls_loss / num_steps
        avg_dom_loss = total_dom_loss / num_steps
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0

        val_loss, val_acc = evaluate(model, val_loader_src, class_criterion)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Train cls loss: {avg_cls_loss:.4f}, "
            f"Train dom loss: {avg_dom_loss:.4f}, "
            f"Train acc (src): {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> New best DANN model saved (Val Acc: {best_val_acc:.4f})")

    print("Training finished.")
    print("Best validation accuracy (source):", best_val_acc)
    print("Best DANN model path:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
