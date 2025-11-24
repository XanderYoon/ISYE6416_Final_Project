# src/separation/mask_separation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from audio.io import load_wav_scipy as load_audio
from .stft_utils import compute_stft, istft, magnitude_spectrogram


@dataclass
class SeparationResult:
    """Container for separated waveforms and masks."""
    heart_est: np.ndarray
    lung_est: np.ndarray
    heart_mask: np.ndarray
    lung_mask: np.ndarray


# ============================================================
#               MASK CONSTRUCTION
# ============================================================

def compute_soft_masks_from_templates(
    stft_mix: np.ndarray,
    heart_template: np.ndarray,
    lung_template: np.ndarray,
    power: float = 2.0,
    eps: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frequency-only soft masks for heart and lung using global templates.
    """
    n_freq, n_frames = stft_mix.shape

    H = heart_template.reshape(-1, 1)  # (F,1)
    L = lung_template.reshape(-1, 1)   # (F,1)

    if H.shape[0] != n_freq or L.shape[0] != n_freq:
        raise ValueError("Template and STFT freq dimensions mismatch.")

    denom = H + L + eps
    heart_mask = H / denom
    lung_mask = L / denom

    heart_mask = np.broadcast_to(heart_mask, (n_freq, n_frames))
    lung_mask  = np.broadcast_to(lung_mask, (n_freq, n_frames))

    return heart_mask.astype(np.float32), lung_mask.astype(np.float32)


# ============================================================
#               MAIN SEPARATION FUNCTION
# ============================================================

def separate_mixture_from_audio(
    audio_mix: np.ndarray,
    sr: int,
    heart_template: np.ndarray,
    lung_template: np.ndarray,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    power: float = 2.0,
) -> SeparationResult:

    stft_mix = compute_stft(audio_mix, sr, n_fft=n_fft, hop_length=hop_length)
    heart_mask, lung_mask = compute_soft_masks_from_templates(
        stft_mix, heart_template, lung_template, power=power
    )

    heart_stft = stft_mix * heart_mask
    lung_stft  = stft_mix * lung_mask

    heart_est = istft(heart_stft, hop_length=hop_length, length=len(audio_mix))
    lung_est  = istft(lung_stft,  hop_length=hop_length, length=len(audio_mix))

    return SeparationResult(
        heart_est=heart_est.astype(np.float32),
        lung_est=lung_est.astype(np.float32),
        heart_mask=heart_mask,
        lung_mask=lung_mask,
    )


# ============================================================
#               METRIC UTILITIES
# ============================================================

def _align_signals(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L = min(len(a), len(b))
    return a[:L], b[:L]


def waveform_mse(ref: np.ndarray, est: np.ndarray) -> float:
    ref_a, est_a = _align_signals(ref, est)
    return float(np.mean((ref_a - est_a) ** 2))


def waveform_corr(ref: np.ndarray, est: np.ndarray) -> float:
    ref_a, est_a = _align_signals(ref, est)
    if np.all(ref_a == 0) or np.all(est_a == 0):
        return 0.0
    return float(np.corrcoef(ref_a, est_a)[0, 1])


def waveform_r2(ref: np.ndarray, est: np.ndarray) -> float:
    """
    R2 score = 1 - SSE/SST
    """
    ref_a, est_a = _align_signals(ref, est)
    sse = np.sum((ref_a - est_a)**2)
    sst = np.sum((ref_a - ref_a.mean())**2)
    if sst == 0:
        return 0.0
    return float(1 - sse/sst)


def spectrogram_mse(
    ref: np.ndarray,
    est: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    power: float = 2.0,
) -> float:
    ref_stft = compute_stft(ref, sr, n_fft=n_fft, hop_length=hop_length)
    est_stft = compute_stft(est, sr, n_fft=n_fft, hop_length=hop_length)

    ref_spec = magnitude_spectrogram(ref_stft, power=power)
    est_spec = magnitude_spectrogram(est_stft, power=power)

    F = min(ref_spec.shape[0], est_spec.shape[0])
    T = min(ref_spec.shape[1], est_spec.shape[1])

    return float(np.mean((ref_spec[:F, :T] - est_spec[:F, :T]) ** 2))


# ============================================================
#               PER-FILE EVALUATION
# ============================================================

def evaluate_single_mixture(
    row: pd.Series,
    heart_template: np.ndarray,
    lung_template: np.ndarray,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    power: float = 2.0,
    mixture_col: str = "mixture_file",
    heart_ref_col: str = "heart_ref_file",
    lung_ref_col: str = "lung_ref_file",
) -> Dict[str, float]:

    mix_path = row[mixture_col]
    heart_ref_path = row[heart_ref_col]
    lung_ref_path = row[lung_ref_col]

    mix_sr, mix_audio = load_audio(mix_path, normalize=True)
    heart_sr, heart_ref = load_audio(heart_ref_path, normalize=True)
    lung_sr, lung_ref = load_audio(lung_ref_path, normalize=True)

    if not (mix_sr == heart_sr == lung_sr):
        raise ValueError("Sampling rate mismatch across files.")

    sep = separate_mixture_from_audio(
        mix_audio, mix_sr,
        heart_template=heart_template,
        lung_template=lung_template,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
    )

    # ---------- Metrics ----------
    mse_h = waveform_mse(heart_ref, sep.heart_est)
    mse_l = waveform_mse(lung_ref,  sep.lung_est)

    corr_h = waveform_corr(heart_ref, sep.heart_est)
    corr_l = waveform_corr(lung_ref,  sep.lung_est)

    r2_h = waveform_r2(heart_ref, sep.heart_est)
    r2_l = waveform_r2(lung_ref,  sep.lung_est)

    spec_mse_h = spectrogram_mse(heart_ref, sep.heart_est, mix_sr)
    spec_mse_l = spectrogram_mse(lung_ref,  sep.lung_est,  mix_sr)

    return {
        "mixed_sound_id": row.get("mixed_sound_id", np.nan),
        "gender": row.get("gender", None),
        "heart_sound_type": row.get("heart_sound_type", None),
        "lung_sound_type": row.get("lung_sound_type", None),
        "location": row.get("location", None),

        "waveform_mse_heart": mse_h,
        "waveform_mse_lung": mse_l,
        "waveform_corr_heart": corr_h,
        "waveform_corr_lung": corr_l,
        "waveform_r2_heart": r2_h,
        "waveform_r2_lung": r2_l,

        "spectrogram_mse_heart": spec_mse_h,
        "spectrogram_mse_lung": spec_mse_l,
    }


# ============================================================
#           BULK EVALUATION OVER MIX.CSV
# ============================================================

def evaluate_over_dataframe(
    mix_df: pd.DataFrame,
    heart_template: np.ndarray,
    lung_template: np.ndarray,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    power: float = 2.0,
    max_rows: Optional[int] = None,
    verbose_every: int = 20,
) -> pd.DataFrame:

    metrics_list: List[Dict[str, float]] = []

    for i, (_, row) in enumerate(mix_df.iterrows()):
        if max_rows is not None and i >= max_rows:
            break

        metrics = evaluate_single_mixture(
            row,
            heart_template=heart_template,
            lung_template=lung_template,
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,
        )
        metrics_list.append(metrics)

    return pd.DataFrame(metrics_list)
