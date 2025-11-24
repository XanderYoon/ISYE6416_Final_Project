# src/separation/em_nmf.py

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

from audio.io import load_wav_scipy as load_audio
from .stft_utils import compute_stft, istft, magnitude_spectrogram
from .hmm_smoothing import hmm_smooth_posteriors

# ---------------------------------------------------------
#  EM-NMF: Latent Variable Probabilistic Factorization
# ---------------------------------------------------------

def em_nmf(V, W_init, H_init, n_iter=60, eps=1e-10):
    """
    Probabilistic latent source EM-NMF.
    V_ft = sum_k W_fk * H_kt

    E-step:
        r_fkt = W_fk * H_kt / sum_j W_fj * H_jt

    M-step:
        W_fk = sum_t r_fkt * V_ft
        H_kt = sum_f r_fkt * V_ft
    """

    W = W_init.copy() + eps     # (F,K)
    H = H_init.copy() + eps     # (K,T)
    F, T = V.shape
    K = W.shape[1]

    for _ in range(n_iter):
        WH = W @ H + eps

        # E-step: responsibilities
        R = (W[:, :, None] * H[None, :, :]) / WH[:, None, :]  # shape (F,K,T)

        # M-step
        W_new = np.sum(R * V[:, None, :], axis=2)     # (F,K)
        H_new = np.sum(R * V[:, None, :], axis=0)     # (K,T)

        # Normalize W to avoid scale degeneracy
        W = W_new / (np.sum(W_new, axis=0, keepdims=True) + eps)
        H = H_new

    return W, H


# ---------------------------------------------------------
#  Apply EM-NMF to mixture & build masks
# ---------------------------------------------------------

@dataclass
class EMNMFResult:
    heart_est: np.ndarray
    lung_est: np.ndarray
    W: np.ndarray
    H: np.ndarray


def em_nmf_separate(
    mix_audio,
    sr,
    heart_template,
    lung_template,
    n_fft=2048,
    hop_length=None,
    n_iter=60
):
    """
    EM-based probabilistic NMF separation using templates as priors.
    """

    # 1) STFT
    S = compute_stft(mix_audio, sr, n_fft=n_fft, hop_length=hop_length)
    V = magnitude_spectrogram(S, power=1.0)  # magnitude, not power

    F, T = V.shape
    K = 2

    # 2) Initialize W from templates
    W_init = np.stack([heart_template, lung_template], axis=1)
    W_init = W_init / np.maximum(W_init.sum(axis=0, keepdims=True), 1e-10)

    # 3) Initialize H randomly
    H_init = np.abs(np.random.randn(K, T)) + 1e-6

    # 4) EM-NMF
    W, H = em_nmf(V, W_init, H_init, n_iter=n_iter)

    # 5) Soft masks
    WH = W @ H + 1e-10
    heart_mask = (W[:, 0][:, None] * H[0][None, :]) / WH
    lung_mask  = (W[:, 1][:, None] * H[1][None, :]) / WH

    # 6) Apply masks to complex STFT
    heart_S = S * heart_mask
    lung_S  = S * lung_mask

    heart_est = istft(heart_S, hop_length=hop_length, length=len(mix_audio))
    lung_est  = istft(lung_S,  hop_length=hop_length, length=len(mix_audio))

    return EMNMFResult(
        heart_est=heart_est.astype(np.float32),
        lung_est=lung_est.astype(np.float32),
        W=W,
        H=H,
    )

def em_nmf_hmm_separate(
    mix_audio,
    sr,
    heart_template,
    lung_template,
    n_fft=2048,
    hop_length=None,
    n_iter=60,
    p_stay: float = 0.99,
):
    """
    EM-NMF separation + HMM temporal smoothing.

    1. Run EM-NMF to get W, H.
    2. Build initial time-frequency masks.
    3. Compute per-frame heart fraction from H.
    4. Run 2-state HMM to smooth heart vs lung dominance over time.
    5. Use posteriors to reweight masks and reconstruct waveforms.
    """

    # --- Step 1: standard EM-NMF separation, but keep masks ---

    # STFT
    S = compute_stft(mix_audio, sr, n_fft=n_fft, hop_length=hop_length)
    V = magnitude_spectrogram(S, power=1.0)

    F, T = V.shape
    K = 2

    # Initialize W from templates
    W_init = np.stack([heart_template, lung_template], axis=1)
    W_init = W_init / np.maximum(W_init.sum(axis=0, keepdims=True), 1e-10)

    # Initialize H randomly
    H_init = np.abs(np.random.randn(K, T)) + 1e-6

    # EM-NMF
    W, H = em_nmf(V, W_init, H_init, n_iter=n_iter)

    # Base masks from NMF
    WH = W @ H + 1e-10
    heart_mask = (W[:, 0][:, None] * H[0][None, :]) / WH
    lung_mask  = (W[:, 1][:, None] * H[1][None, :]) / WH

    # --- Step 2: compute per-frame heart fraction from H ---

    # We treat H[0,t] vs H[1,t] as heart vs lung activation strength
    heart_act = H[0]
    lung_act  = H[1]
    denom = heart_act + lung_act + 1e-10
    heart_fraction = heart_act / denom  # shape (T,)

    # --- Step 3: HMM smoothing over time ---

    gamma_heart = hmm_smooth_posteriors(heart_fraction, p_stay=p_stay)  # (T,)
    gamma_lung = 1.0 - gamma_heart

    # --- Step 4: reweight masks using HMM posteriors ---

    gamma_heart_2d = gamma_heart[None, :]   # (1,T) for broadcasting
    gamma_lung_2d  = gamma_lung[None, :]

    heart_mask_smooth = heart_mask * gamma_heart_2d
    lung_mask_smooth  = lung_mask * gamma_lung_2d

    den = heart_mask_smooth + lung_mask_smooth + 1e-10
    heart_mask_smooth /= den
    lung_mask_smooth  /= den

    # --- Step 5: reconstruct waveforms ---

    heart_S = S * heart_mask_smooth
    lung_S  = S * lung_mask_smooth

    heart_est = istft(heart_S, hop_length=hop_length, length=len(mix_audio))
    lung_est  = istft(lung_S,  hop_length=hop_length, length=len(mix_audio))

    return EMNMFResult(
        heart_est=heart_est.astype(np.float32),
        lung_est=lung_est.astype(np.float32),
        W=W,
        H=H,
    )


# ---------------------------------------------------------
#   Evaluation only (metrics required for Stage 3)
# ---------------------------------------------------------

def _trim(a, b):
    L = min(len(a), len(b))
    return a[:L], b[:L]


def mse(a, b):
    a2, b2 = _trim(a, b)
    return float(np.mean((a2 - b2) ** 2))


def corr(a, b):
    a2, b2 = _trim(a, b)
    if np.std(a2) < 1e-8 or np.std(b2) < 1e-8:
        return 0.0
    return float(np.corrcoef(a2, b2)[0, 1])


def spec_mse(a, b, sr, n_fft=2048, hop_length=None):
    Sa = magnitude_spectrogram(compute_stft(a, sr, n_fft, hop_length))
    Sb = magnitude_spectrogram(compute_stft(b, sr, n_fft, hop_length))

    F = min(Sa.shape[0], Sb.shape[0])
    T = min(Sa.shape[1], Sb.shape[1])
    return float(np.mean((Sa[:F, :T] - Sb[:F, :T]) ** 2))

def r2(a, b):
    """
    Coefficient of determination (R^2) between reference (a) and estimate (b).
    """
    a2, b2 = _trim(a, b)

    ss_res = np.sum((a2 - b2) ** 2)
    ss_tot = np.sum((a2 - np.mean(a2)) ** 2)

    if ss_tot < 1e-12:
        return 0.0

    return float(1 - ss_res / ss_tot)
