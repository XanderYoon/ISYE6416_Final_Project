import os
from typing import Tuple

import librosa
import numpy as np
from scipy.io import wavfile


def load_wav_scipy(path: str, normalize: bool = True) -> Tuple[int, np.ndarray]:
    """
    Load a WAV file using scipy and optionally normalize to [-1, 1].

    Returns
    -------
    sample_rate : int
    audio       : np.ndarray, shape (T,), dtype float32
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    sample_rate, data = wavfile.read(path)

    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # Convert stereo to mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    if normalize:
        max_val = float(np.max(np.abs(data)))
        if max_val > 0:
            data = data / max_val

    return sample_rate, data


def load_audio_mono(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono using librosa with a consistent sampling rate.

    Parameters
    ----------
    path : str
        Path to .wav file.
    target_sr : int
        Target sampling rate for resampling.

    Returns
    -------
    audio : np.ndarray, shape (T,)
    sample_rate : int
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr
