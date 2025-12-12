import os
import torch
import torch.nn as nn
from torch.optim import Adam

# Use absolute package import to ensure the sibling package is found.
from src.data.iemocap_dataset_loader import load_iemocap_metadata
from src.baseline.dataloaders import create_dataloaders
from src.baseline.model_cnn import EmotionCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = r"D:\Recordings\Science\DL\IEMOCAP_full_release"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for mel, labels in loader:
        mel = mel.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(mel)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mel.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for mel, labels in loader:
            mel = mel.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(mel)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * mel.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    print("Using device:", DEVICE)

    # Load all samples
    samples = load_iemocap_metadata(BASE_PATH)

    # Group samples by session - 60/20/20 split
    session_to_samples = {}
    for s in samples:
        sess = s["session"]          # "Session1" ... "Session5"
        session_to_samples.setdefault(sess, []).append(s)

    # Speaker-independent split:
    train_sessions = ["Session1", "Session2", "Session3"]
    val_sessions   = ["Session4"]
    test_sessions  = ["Session5"]

    train_samples = []
    val_samples = []
    test_samples = []

    for sess in train_sessions:
        train_samples.extend(session_to_samples[sess])

    for sess in val_sessions:
        val_samples.extend(session_to_samples[sess])

    for sess in test_sessions:
        test_samples.extend(session_to_samples[sess])

    print("Train sessions:", train_sessions, "→", len(train_samples), "samples")
    print("Val sessions:  ", val_sessions,   "→", len(val_samples), "samples")
    print("Test sessions: ", test_sessions,  "→", len(test_samples), "samples")

    # Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples, val_samples, test_samples, batch_size=16
    )

    # Model, loss, optimizer
    model = EmotionCNN(num_classes=5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    EPOCHS = 20  # you can change this later
    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "best_emotion_cnn.pt")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (Val Acc: {best_val_acc:.4f})")

    print("Training finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model path: {best_model_path}")


if __name__ == "__main__":
    main()
