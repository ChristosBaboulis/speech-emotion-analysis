# Wav2Vec2-based emotion recognition models with Domain Adaptation (DANN)
#
# This module contains pre-trained Wav2Vec2-based models with Domain Adversarial Neural Networks (DANN)
# for speech emotion recognition. Uses Facebook's Wav2Vec2 pre-trained encoder with fine-tuning
# capability on raw audio waveforms, extended with domain adaptation for better generalization
# from source domain (IEMOCAP) to target domain (RAVDESS).
#
# Main components:
# - model_wav2vec_dann.py: Wav2Vec2 encoder + emotion classifier + domain classifier (DANN)
# - train_wav2vec_dann.py: Fine-tuning training script with domain adaptation and mixed precision
# - eval_iemocap_wav2vec_dann.py: Evaluation script for IEMOCAP test set (source domain)
# - eval_ravdess_wav2vec_dann.py: Evaluation script for RAVDESS test set (target domain)
# - audio_dataset.py: Dataset class for raw audio loading (replaces mel-spectrogram)
# - dataloaders.py: Data loading utilities for standard evaluation (IEMOCAP train/val/test)
# - dann_dataloaders.py: Data loading utilities for domain adaptation (IEMOCAP source + RAVDESS target)
