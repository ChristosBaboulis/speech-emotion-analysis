import os
from dataset_loader import load_iemocap_metadata

BASE_PATH = r"D:\Recordings\Science\DL\IEMOCAP_full_release"

print("BASE_PATH exists?", os.path.isdir(BASE_PATH))
for sess in range(1, 6):
    session = f"Session{sess}"
    session_path = os.path.join(BASE_PATH, session)
    print(session, "->", session_path, "exists?", os.path.isdir(session_path))

samples = load_iemocap_metadata(BASE_PATH)

print("Total samples:", len(samples))
print(samples[0] if samples else "No samples found")
