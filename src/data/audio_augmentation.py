"""
Audio augmentation utilities for speech emotion recognition.
Implements time stretching, pitch shifting, and noise injection.
"""

import numpy as np
import librosa
import torch


def time_stretch(y: np.ndarray, sr: int, rate: float = None) -> np.ndarray:
    """
    Apply time stretching (speed perturbation) to audio.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        rate: Stretch factor (None = random between 0.85 and 1.15)
    
    Returns:
        Stretched audio waveform
    """
    if rate is None:
        rate = np.random.uniform(0.85, 1.15)  # ±15% speed variation
    
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float = None) -> np.ndarray:
    """
    Apply pitch shifting to audio.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        n_steps: Number of semitones to shift (None = random between -2 and +2)
    
    Returns:
        Pitch-shifted audio waveform
    """
    if n_steps is None:
        n_steps = np.random.uniform(-2.0, 2.0)  # ±2 semitones
    
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def add_noise(y: np.ndarray, noise_factor: float = None) -> np.ndarray:
    """
    Add Gaussian noise to audio.
    
    Args:
        y: Audio waveform
        noise_factor: Noise level (None = random between 0.01 and 0.05)
    
    Returns:
        Noisy audio waveform
    """
    if noise_factor is None:
        noise_factor = np.random.uniform(0.01, 0.05)  # 1-5% noise
    
    noise = np.random.normal(0, noise_factor, y.shape).astype(y.dtype)
    return y + noise


def apply_augmentation(y: np.ndarray, sr: int, p: float = 0.5) -> np.ndarray:
    """
    Apply random augmentation to audio with probability p.
    
    Augmentation techniques (applied independently):
    - Time stretching: probability p
    - Pitch shifting: probability p
    - Noise injection: probability p
    
    Args:
        y: Audio waveform
        sr: Sample rate
        p: Probability of applying each augmentation
    
    Returns:
        Augmented audio waveform
    """
    augmented = y.copy()
    
    # Time stretching
    if np.random.rand() < p:
        augmented = time_stretch(augmented, sr)
    
    # Pitch shifting
    if np.random.rand() < p:
        augmented = pitch_shift(augmented, sr)
    
    # Noise injection
    if np.random.rand() < p:
        augmented = add_noise(augmented)
    
    return augmented

