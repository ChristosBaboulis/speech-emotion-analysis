# Data loading and preprocessing utilities
#
# This module contains dataset loaders and preprocessing utilities for speech emotion recognition.
# Supports IEMOCAP and RAVDESS datasets with consistent label mapping and preprocessing pipelines.
#
# Main components:
# - iemocap_dataset_loader.py: IEMOCAP dataset metadata loading and session-based splitting
# - iemocap_dataset.py: IEMOCAP dataset class with mel-spectrogram preprocessing
# - ravdess_dataset_loader.py: RAVDESS dataset metadata loading with emotion label mapping
# - ravdess_dataset.py: RAVDESS dataset class with mel-spectrogram preprocessing
#
# Label mapping (consistent across datasets):
# - angry: 0, frustrated: 1, happy: 2, neutral: 3, sad: 4

