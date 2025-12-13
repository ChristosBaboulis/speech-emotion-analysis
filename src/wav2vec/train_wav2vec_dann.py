import os
import warnings
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.amp import autocast, GradScaler
import math

# Suppress transformers deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from src.data.iemocap_dataset_loader import load_iemocap_metadata, split_iemocap_by_sessions
from src.wav2vec.dataloaders import create_wav2vec_dataloaders
from src.wav2vec.dann_dataloaders import create_wav2vec_dann_loaders
from src.wav2vec.model_wav2vec_dann import Wav2VecDANNEmotionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IEMOCAP_BASE = r"D:\Recordings\Science\DL\IEMOCAP_full_release"
RAVDESS_BASE = r"D:\Recordings\Science\DL\RAVDESS"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_wav2vec_dann_emotion.pt")

# Hyperparameters
NUM_EPOCHS = 40
BATCH_SIZE = 4  # Smaller batch size for Wav2Vec2 memory requirements
LR_ENCODER = 5e-5  # Lower LR for pre-trained encoder
LR_CLASSIFIER = 1e-3  # Higher LR for classifier heads
LAMBDA_DOMAIN = 0.5
MAX_AUDIO_LENGTH = 10.0  # 10 seconds max


def evaluate(model, loader, criterion):
    """Evaluate model on validation set (source domain only)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

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

            with autocast(device_type='cuda'):
                emotion_logits, _ = model(audios, attention_mask=attention_masks, alpha=0.0)
                loss = criterion(emotion_logits, labels)

            total_loss += loss.item() * audios.size(0)
            preds = emotion_logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_acc


def main():
    print("Using device:", DEVICE)
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ---------- Load IEMOCAP and make speaker-independent split ----------
    samples = load_iemocap_metadata(IEMOCAP_BASE)
    train_samples, val_samples, test_samples = split_iemocap_by_sessions(samples)

    print("Train samples:", len(train_samples))
    print("Val samples:  ", len(val_samples))
    print("Test samples: ", len(test_samples))

    # Dataloaders for evaluation (source domain only)
    _, val_loader_src, _ = create_wav2vec_dataloaders(
        train_samples, val_samples, test_samples,
        batch_size=BATCH_SIZE,
        max_audio_length=MAX_AUDIO_LENGTH
    )

    # ---------- DANN loaders (source: IEMOCAP train, target: RAVDESS) ----------
    src_loader, tgt_loader = create_wav2vec_dann_loaders(
        train_samples,  # Pre-split IEMOCAP training samples (Session1-3)
        RAVDESS_BASE,
        batch_size=BATCH_SIZE,
        max_audio_length=MAX_AUDIO_LENGTH,
        num_workers=2,
    )

    print("Source DANN batches:", len(src_loader))
    print("Target DANN batches:", len(tgt_loader))

    # ---------- Model, losses, optimizer ----------
    print("\nLoading Wav2Vec2 DANN model (this may download on first run)...")
    model = Wav2VecDANNEmotionModel(num_classes=5, num_domains=2).to(DEVICE)
    
    # Separate learning rates for encoder and classifier heads
    encoder_params = list(model.wav2vec.parameters())
    classifier_params = list(model.classifier.parameters()) + list(model.domain_head.parameters())
    
    optimizer = Adam(
        [
            {'params': encoder_params, 'lr': LR_ENCODER},
            {'params': classifier_params, 'lr': LR_CLASSIFIER}
        ]
    )
    
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device='cuda')  # For mixed precision

    best_val_acc = 0.0

    print(f"\nTraining for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE}")
    print(f"Encoder LR: {LR_ENCODER}, Classifier LR: {LR_CLASSIFIER}, Lambda Domain: {LAMBDA_DOMAIN}")
    print("-" * 70)

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
            # Get source batch
            try:
                src_batch = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                src_batch = next(src_iter)

            # Get target batch
            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_batch = next(tgt_iter)

            # Unpack batches (audio, label, attention_mask, domain_id) from DANN loaders
            src_audios, src_labels, src_masks, src_domains = src_batch
            tgt_audios, tgt_labels, tgt_masks, tgt_domains = tgt_batch  # tgt_labels unused

            src_audios = src_audios.to(DEVICE)
            src_labels = src_labels.to(DEVICE)
            src_domains = src_domains.to(DEVICE)
            if src_masks is not None:
                src_masks = src_masks.to(DEVICE)

            tgt_audios = tgt_audios.to(DEVICE)
            tgt_domains = tgt_domains.to(DEVICE)
            if tgt_masks is not None:
                tgt_masks = tgt_masks.to(DEVICE)

            # Concatenate for domain classification
            all_audios = torch.cat([src_audios, tgt_audios], dim=0)
            all_domains = torch.cat([src_domains, tgt_domains], dim=0)
            if src_masks is not None and tgt_masks is not None:
                all_masks = torch.cat([src_masks, tgt_masks], dim=0)
            else:
                all_masks = None

            # Alpha scheduling for gradient reversal (standard DANN approach)
            # Progress from 0 to 1 across training: p in [0, 1]
            p = (epoch - 1 + step / num_steps) / NUM_EPOCHS
            # Sigmoid-like schedule: alpha increases smoothly from 0 to MAX_ALPHA
            alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0  # [0, ~1] for p in [0, 1]
            MAX_ALPHA = 0.7  # Cap alpha to prevent too strong domain alignment early
            alpha = min(alpha, MAX_ALPHA)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                emotion_logits, domain_logits = model(all_audios, attention_mask=all_masks, alpha=alpha)

                # Classification loss only on source samples
                bs_src = src_audios.size(0)
                cls_logits = emotion_logits[:bs_src]

                loss_cls = class_criterion(cls_logits, src_labels)
                loss_dom = domain_criterion(domain_logits, all_domains)
                loss = loss_cls + LAMBDA_DOMAIN * loss_dom

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Stats
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
            print(f"  -> New best Wav2Vec2 DANN model saved (Val Acc: {best_val_acc:.4f})")

    print("\nTraining finished.")
    print("Best validation accuracy (source):", best_val_acc)
    print("Best Wav2Vec2 DANN model path:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()

