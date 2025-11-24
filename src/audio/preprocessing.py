from typing import Tuple

import librosa
import numpy as np


def trim_silence(audio: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    """
    Trim leading/trailing silence from an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal.
    top_db : float
        Threshold in decibels below reference to consider as silence.

    Returns
    -------
    trimmed : np.ndarray
        Audio signal with silence removed.
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def resample_audio(audio: np.ndarray, sample_rate: int, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Uniformly resample audio to target sample rate.

    Parameters
    ----------
    audio : np.ndarray
    sample_rate : int
    target_sr : int

    Returns
    -------
    resampled_audio : np.ndarray
    target_sr       : int
    """
    if sample_rate == target_sr:
        return audio, sample_rate

    resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
    return resampled, target_sr
