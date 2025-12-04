import torch
import torch.nn as nn
from torch.optim import Adam

from dataset_loader import load_iemocap_metadata
from sklearn.model_selection import train_test_split
from dataloaders import create_dataloaders
from model_cnn import EmotionCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = r"D:\Recordings\Science\DL\IEMOCAP_full_release"


def main():
    print("Using device:", DEVICE)

    # Load all samples
    samples = load_iemocap_metadata(BASE_PATH)

    # Split 70/15/15
    train_samples, temp = train_test_split(samples, test_size=0.30, shuffle=True, random_state=42)
    val_samples, test_samples = train_test_split(temp, test_size=0.50, shuffle=True, random_state=42)

    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples, val_samples, test_samples, batch_size=16
    )

    # Model
    model = EmotionCNN(num_classes=5).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    EPOCHS = 3  # low for first test

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for mel, labels in train_loader:
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

        train_loss = running_loss / total
        train_acc = correct / total

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

    print("Training done.")


if __name__ == "__main__":
    main()
