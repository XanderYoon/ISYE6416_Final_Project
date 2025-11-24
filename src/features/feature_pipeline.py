from typing import Dict

import numpy as np

from audio.preprocessing import trim_silence
from .time_domain import compute_time_domain_features
from .spectral import compute_spectral_features
from .mfcc import compute_mfcc_summary
from .wavelet import compute_wavelet_energy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def extract_all_features(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """
    End-to-end feature extraction pipeline from a single waveform.

    Steps
    -----
    1. Trim silence.
    2. Time-domain features (RMS, ZCR, duration).
    3. Spectral features (centroid, bandwidth, rolloff).
    4. MFCC mean/std per coefficient.
    5. Wavelet band energy features.

    Returns
    -------
    flat_features : Dict[str, float]
        Flattened dictionary of scalar features.
    """
    trimmed = trim_silence(audio)

    features: Dict[str, float] = {}
    features.update(compute_time_domain_features(trimmed, sample_rate))
    features.update(compute_spectral_features(trimmed, sample_rate))

    # MFCCs
    mfcc_summary = compute_mfcc_summary(trimmed, sample_rate)
    for i, value in enumerate(mfcc_summary["mfcc_mean"]):
        features[f"mfcc{i + 1}_mean"] = float(value)
    for i, value in enumerate(mfcc_summary["mfcc_std"]):
        features[f"mfcc{i + 1}_std"] = float(value)

    # Wavelet features
    wavelet_energy = compute_wavelet_energy(trimmed)
    for i, band_energy in enumerate(wavelet_energy):
        features[f"wavelet_energy_{i}"] = float(band_energy)

    return features

def plot_feature_correlation(df, title="Correlation Matrix", figsize=(14, 10)):
    """
    Plots a correlation matrix using your required color scheme.
    Only numerical features are included.
    """

    # Keep only numerical columns
    num_df = df.select_dtypes(include=[np.number])

    corr = num_df.corr().abs()   # absolute correlations (more interpretable)
    vmax = corr.values.max()

    fig, ax = plt.subplots(figsize=figsize)

    # Use imshow to exactly match your color specification
    im = ax.imshow(
        corr.values,
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=vmax,
    )

    # Tick labels
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    plt.title(title, fontsize=14)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.set_ylabel("Correlation", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()
