# DANN (Domain Adversarial Neural Network) models
#
# This module contains the DANN architecture for domain adaptation in speech emotion recognition.
# The model extends the baseline architecture with domain adaptation via gradient reversal layer,
# enabling better generalization from source domain (IEMOCAP) to target domain (RAVDESS).
#
# Main components:
# - model_dann.py: DANN architecture with gradient reversal layer
# - train_dann.py: Training script with domain adaptation
# - eval_iemocap_dann.py: Evaluation script for IEMOCAP test set
# - eval_ravdess_dann.py: Evaluation script for RAVDESS test set (target domain)
# - dann_dataloaders.py: Data loading utilities for domain adaptation

