from torch.utils.data import DataLoader, WeightedRandomSampler
from src.wav2vec.audio_dataset import AudioDataset
import numpy as np
import torch


def collate_fn(batch):
    """
    Custom collate function for variable-length audio sequences.
    Pads sequences to the longest sequence in the batch.
    """
    # Check if domain_id is present (DANN case)
    has_domain_id = len(batch[0]) == 4
    
    if has_domain_id:
        # DANN format: (audio, label, attention_mask, domain_id)
        audios = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        attention_masks = [item[2] for item in batch]
        domain_ids = torch.tensor([item[3] for item in batch], dtype=torch.long)
        has_attention_mask = True
    else:
        # Standard format: (audio, label, attention_mask) or (audio, label)
        audios = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        if len(batch[0]) >= 3:
            attention_masks = [item[2] for item in batch]
            has_attention_mask = True
        else:
            attention_masks = None
            has_attention_mask = False
        domain_ids = None
    
    # Find max length in batch
    max_len = max(audio.shape[0] for audio in audios)
    
    # Pad all audios to max_len
    padded_audios = []
    padded_masks = []
    
    for i, audio in enumerate(audios):
        current_len = audio.shape[0]
        pad_len = max_len - current_len
        
        # Pad audio
        padded_audio = torch.nn.functional.pad(audio, (0, pad_len))
        padded_audios.append(padded_audio)
        
        # Pad attention mask
        if has_attention_mask:
            mask = attention_masks[i]
            if mask.shape[0] < max_len:
                pad_mask = torch.zeros(max_len - mask.shape[0], dtype=torch.long)
                padded_mask = torch.cat([mask, pad_mask])
            else:
                padded_mask = mask
            padded_masks.append(padded_mask)
    
    # Stack into tensors
    batched_audios = torch.stack(padded_audios)
    
    if has_attention_mask:
        batched_masks = torch.stack(padded_masks)
    else:
        batched_masks = None
    
    # Return based on what's available
    if has_domain_id:
        return batched_audios, labels, batched_masks, domain_ids
    elif batched_masks is not None:
        return batched_audios, labels, batched_masks
    else:
        return batched_audios, labels


def create_wav2vec_dataloaders(train_samples, val_samples, test_samples, batch_size=4, max_audio_length=10.0):
    """
    Create dataloaders for Wav2Vec2 training.
    
    Args:
        train_samples: Training samples
        val_samples: Validation samples
        test_samples: Test samples
        batch_size: Batch size (smaller for Wav2Vec2 due to memory)
        max_audio_length: Maximum audio length in seconds
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Class balancing for training
    labels = np.array([s["label"] for s in train_samples])
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Datasets
    train_dataset = AudioDataset(train_samples, max_length=max_audio_length)
    val_dataset = AudioDataset(val_samples, max_length=max_audio_length)
    test_dataset = AudioDataset(test_samples, max_length=max_audio_length)
    
    # Loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,  # Reduced for memory
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader

