from typing import Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 64,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute mel-spectrogram in dB.
    """
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        fmax=sample_rate // 2,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def plot_mel_spectrogram(
    mel_db: np.ndarray,
    sample_rate: int,
    title: str = "Mel Spectrogram",
    hop_length: int = 512,
    cmap: str = "magma",
    figsize: Tuple[int, int] = (10, 4),
) -> None:
    """
    Plot a mel spectrogram.
    """
    plt.figure(figsize=figsize)
    librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap=cmap,
    )
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
