from typing import Dict

import librosa
import numpy as np


def compute_spectral_features(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """
    Compute basic spectral features using librosa: centroid, bandwidth, rolloff.
    """
    stft_mag = np.abs(librosa.stft(audio))
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=stft_mag, sr=sample_rate)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=stft_mag, sr=sample_rate)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=stft_mag, sr=sample_rate)))

    return {
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
    }
