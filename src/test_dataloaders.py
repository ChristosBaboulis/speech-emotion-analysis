from dataset_loader import load_iemocap_metadata
from dataloaders import create_dataloaders
from sklearn.model_selection import train_test_split

BASE_PATH = r"D:\Recordings\Science\DL\IEMOCAP_full_release"

# Load all samples
samples = load_iemocap_metadata(BASE_PATH)

# 70/15/15 split
train_samples, temp = train_test_split(samples, test_size=0.30, shuffle=True, random_state=42)
val_samples, test_samples = train_test_split(temp, test_size=0.50, shuffle=True, random_state=42)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_samples, val_samples, test_samples, batch_size=8
)

# Take one batch from train loader
mel_batch, label_batch = next(iter(train_loader))

print("Mel batch shape:", mel_batch.shape)   # [B, 1, 128, T]
print("Label batch shape:", label_batch.shape)
print("Labels:", label_batch)
