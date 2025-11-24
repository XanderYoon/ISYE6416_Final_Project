# src/separation/stft_utils.py

from typing import Optional, Tuple

import numpy as np
import librosa


def compute_stft(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
) -> np.ndarray:
    """
    Compute complex STFT of a 1D audio signal.

    Parameters
    ----------
    audio : np.ndarray
        1D waveform.
    sr : int
        Sample rate (Hz).
    n_fft : int
        FFT size.
    hop_length : int, optional
        Hop length. If None, defaults to n_fft // 4.
    window : str
        Window type for STFT.
    center : bool
        Whether to center frames.

    Returns
    -------
    stft_matrix : np.ndarray
        Complex STFT with shape (n_freq, n_frames).
    """
    if hop_length is None:
        hop_length = n_fft // 4

    stft_matrix = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=center,
    )
    return stft_matrix


def istft(
    stft_matrix: np.ndarray,
    hop_length: Optional[int] = None,
    window: str = "hann",
    length: Optional[int] = None,
    center: bool = True,
) -> np.ndarray:
    """
    Inverse STFT back to time-domain waveform.

    Parameters
    ----------
    stft_matrix : np.ndarray
        Complex STFT (n_freq, n_frames).
    hop_length : int, optional
        Hop length. If None, defaults to n_fft // 4.
    window : str
        Window type for iSTFT.
    length : int, optional
        Target output length. If provided, output is trimmed/padded.
    center : bool
        Whether the input STFT was computed with center=True.

    Returns
    -------
    audio : np.ndarray
        Reconstructed waveform.
    """
    n_fft = (stft_matrix.shape[0] - 1) * 2
    if hop_length is None:
        hop_length = n_fft // 4

    audio = librosa.istft(
        stft_matrix,
        hop_length=hop_length,
        window=window,
        length=length,
        center=center,
    )
    return audio


def magnitude_spectrogram(stft_matrix: np.ndarray, power: float = 2.0) -> np.ndarray:
    """
    Convert complex STFT to magnitude spectrogram with given power.

    Parameters
    ----------
    stft_matrix : np.ndarray
        Complex STFT.
    power : float
        Power exponent (e.g., 1 for magnitude, 2 for power).

    Returns
    -------
    spec : np.ndarray
        Non-negative spectrogram with shape (n_freq, n_frames).
    """
    return np.abs(stft_matrix) ** power


def db_spectrogram(stft_matrix: np.ndarray, power: float = 2.0) -> np.ndarray:
    """
    Convert complex STFT to dB-scaled magnitude spectrogram.

    Parameters
    ----------
    stft_matrix : np.ndarray
        Complex STFT.
    power : float
        Power exponent prior to dB scaling.

    Returns
    -------
    spec_db : np.ndarray
        dB spectrogram (float32).
    """
    mag = magnitude_spectrogram(stft_matrix, power=power)
    return librosa.power_to_db(mag, ref=np.max)
