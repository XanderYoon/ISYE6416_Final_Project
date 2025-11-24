# src/separation/templates.py

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import librosa

from audio.io import load_wav_scipy as load_audio
from .stft_utils import compute_stft, magnitude_spectrogram


def _template_from_files(
    file_paths: Sequence[str],
    n_fft: int = 2048,
    hop_length: int | None = None,
    power: float = 2.0,
    max_files: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a 1D frequency template from a collection of single-source recordings.

    For each file:
      - compute STFT
      - compute magnitude^power
      - average over time to get a per-frequency vector
    Then average across files, and normalize to sum to 1.

    Parameters
    ----------
    file_paths : sequence of str
        Paths to WAV files for one source (heart-only or lung-only).
    n_fft : int
        FFT size.
    hop_length : int or None
        Hop length for STFT (defaults to n_fft // 4).
    power : float
        Spectrogram power exponent.
    max_files : int or None
        If provided, only use up to this many files (for speed).

    Returns
    -------
    template : np.ndarray
        1D frequency template of shape (n_freq,) normalized to sum 1.
    freqs : np.ndarray
        Frequency axis in Hz, shape (n_freq,).
    """
    per_file_specs: List[np.ndarray] = []
    sample_rate: int | None = None

    if max_files is not None:
        file_paths = file_paths[:max_files]

    for path in file_paths:
        sr, audio = load_audio(path, normalize=True)

        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: {sr} vs {sample_rate}")

        stft_matrix = compute_stft(audio, sr, n_fft=n_fft, hop_length=hop_length)
        spec = magnitude_spectrogram(stft_matrix, power=power)  # (F, T)

        # Average over time axis -> (F,)
        freq_profile = spec.mean(axis=1)
        per_file_specs.append(freq_profile)

    if not per_file_specs:
        raise ValueError("No valid files provided for template building.")

    stacked = np.stack(per_file_specs, axis=0)  # (n_files, F)
    template = stacked.mean(axis=0)  # (F,)

    # Normalize to sum to 1 (avoid division by zero)
    template = np.maximum(template, 1e-12)
    template = template / template.sum()

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    return template.astype(np.float32), freqs.astype(np.float32)


def build_heart_lung_templates(
    heart_df: pd.DataFrame,
    lung_df: pd.DataFrame,
    heart_filename_col: str = "filename",
    lung_filename_col: str = "filename",
    n_fft: int = 2048,
    hop_length: int | None = None,
    power: float = 2.0,
    max_files_per_source: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build heart and lung templates from their respective single-source metadata.

    Parameters
    ----------
    heart_df : pd.DataFrame
        DataFrame for heart-only recordings. Must contain `heart_filename_col`.
    lung_df : pd.DataFrame
        DataFrame for lung-only recordings. Must contain `lung_filename_col`.
    heart_filename_col : str
        Column in `heart_df` with full relative paths to heart WAVs.
    lung_filename_col : str
        Column in `lung_df` with full relative paths to lung WAVs.
    n_fft : int
        FFT size.
    hop_length : int or None
        Hop length.
    power : float
        Spectrogram power exponent.
    max_files_per_source : int or None
        Optional limit on number of files for each template.

    Returns
    -------
    heart_template : np.ndarray
        Heart frequency template, shape (n_freq,).
    lung_template : np.ndarray
        Lung frequency template, shape (n_freq,).
    freqs : np.ndarray
        Frequency axis in Hz, shared for both templates.
    """
    heart_paths = heart_df[heart_filename_col].tolist()
    lung_paths = lung_df[lung_filename_col].tolist()

    heart_template, freqs_h = _template_from_files(
        heart_paths,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        max_files=max_files_per_source,
    )

    lung_template, freqs_l = _template_from_files(
        lung_paths,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        max_files=max_files_per_source,
    )

    if heart_template.shape != lung_template.shape:
        raise ValueError(
            f"Template shapes do not match: heart={heart_template.shape}, lung={lung_template.shape}"
        )

    if not np.allclose(freqs_h, freqs_l):
        raise ValueError("Heart and lung templates have different frequency axes.")

    return heart_template, lung_template, freqs_h
