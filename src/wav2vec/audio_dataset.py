import torch
from torch.utils.data import Dataset
import librosa
import numpy as np


class AudioDataset(Dataset):
    """
    Dataset for raw audio loading (for Wav2Vec2).
    Returns raw waveforms instead of mel-spectrograms.
    """

    def __init__(self, samples, sr=16000, max_length=None, domain_id=None):
        """
        Args:
            samples: List of sample dicts with 'path' and 'label'
            sr: Sample rate (Wav2Vec2 expects 16kHz)
            max_length: Maximum audio length in seconds (None = no limit)
            domain_id: Optional domain ID for DANN
        """
        self.samples = samples
        self.sr = sr
        self.max_length = max_length  # in seconds
        self.max_samples = int(max_length * sr) if max_length else None
        self.domain_id = domain_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        wav_path = item["path"]
        label = item["label"]

        # Load raw audio
        y, sr = librosa.load(wav_path, sr=self.sr, mono=True)

        # Convert to tensor
        audio = torch.tensor(y, dtype=torch.float32)

        # Pad or truncate to max_length if specified
        if self.max_samples is not None:
            current_len = audio.shape[0]
            if current_len > self.max_samples:
                audio = audio[:self.max_samples]
            elif current_len < self.max_samples:
                pad_len = self.max_samples - current_len
                audio = torch.nn.functional.pad(audio, (0, pad_len))

        # Create attention mask (1 for actual audio, 0 for padding)
        if self.max_samples is not None:
            attention_mask = torch.ones(len(audio), dtype=torch.long)
            if audio.shape[0] < self.max_samples:
                # Pad attention mask
                pad_mask = torch.zeros(self.max_samples - audio.shape[0], dtype=torch.long)
                attention_mask = torch.cat([attention_mask, pad_mask])
        else:
            attention_mask = torch.ones(len(audio), dtype=torch.long)

        if self.domain_id is not None:
            return audio, label, attention_mask, self.domain_id
        else:
            return audio, label, attention_mask


if __name__ == "__main__":
    # Quick test
    from src.data.iemocap_dataset_loader import load_iemocap_metadata
    
    BASE_PATH = r"D:\Recordings\Science\DL\IEMOCAP_full_release"
    samples = load_iemocap_metadata(BASE_PATH)
    
    dataset = AudioDataset(samples[:10], max_length=10.0)  # 10 seconds max
    audio, label, mask = dataset[0]
    
    print(f"Audio shape: {audio.shape}")
    print(f"Label: {label}")
    print(f"Attention mask shape: {mask.shape}")
    print(f"Audio duration: {len(audio) / 16000:.2f} seconds")

