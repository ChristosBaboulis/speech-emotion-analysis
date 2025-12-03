import os
from sklearn.model_selection import train_test_split
from dataset_loader import load_iemocap_metadata

BASE_PATH = r"D:\Recordings\Science\DL\IEMOCAP_full_release"

print("BASE_PATH exists?", os.path.isdir(BASE_PATH))
for sess in range(1, 6):
    session = f"Session{sess}"
    session_path = os.path.join(BASE_PATH, session)
    print(session, "->", session_path, "exists?", os.path.isdir(session_path))

# Load IEMOCAP samples
samples = load_iemocap_metadata(BASE_PATH)

print("Total samples:", len(samples))

# 70% train, 15% val, 15% test
train_samples, temp = train_test_split(
    samples, test_size=0.30, shuffle=True, random_state=42
)
val_samples, test_samples = train_test_split(
    temp, test_size=0.50, shuffle=True, random_state=42
)

print("Train samples:", len(train_samples))
print("Val samples:", len(val_samples))
print("Test samples:", len(test_samples))

# Example
print("Example train sample:", train_samples[0] if train_samples else "No train samples")
