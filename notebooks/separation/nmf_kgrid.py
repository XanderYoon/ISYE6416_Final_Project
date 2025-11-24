# src/separation/nmf_kgrid.py

import numpy as np
from typing import List, Dict
from audio.io import load_wav_scipy as load_audio

from .stft_utils import compute_stft, istft, magnitude_spectrogram
from .em_nmf import em_nmf       # reuse your EM-NMF core
from .em_nmf import mse, corr, r2, spec_mse


# ---------------------------------------------------------
# Utility: cosine similarity between W[:,k] and template
# ---------------------------------------------------------
def cosine_sim(a, b, eps=1e-12):
    a_norm = a / (np.linalg.norm(a) + eps)
    b_norm = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a_norm, b_norm))


# ---------------------------------------------------------
# Main separation for K components
# ---------------------------------------------------------
def em_nmf_k_components(
    mix_audio,
    sr,
    heart_template,
    lung_template,
    K=4,
    n_fft=2048,
    hop_length=None,
    n_iter=60,
):
    """
    EM-NMF with K components,
    then assign each component to heart or lung using cosine
    similarity with templates.
    """

    # STFT and magnitude
    S = compute_stft(mix_audio, sr, n_fft=n_fft, hop_length=hop_length)
    V = magnitude_spectrogram(S, power=1.0)  # (F,T)

    F, T = V.shape

    # Initialize W: template-informed for two columns, random for others
    W_init = np.random.rand(F, K)
    W_init[:, 0] = heart_template
    W_init[:, 1] = lung_template
    W_init = W_init / np.maximum(W_init.sum(axis=0, keepdims=True), 1e-10)

    # Random H
    H_init = np.abs(np.random.randn(K, T)) + 1e-6

    # EM-NMF
    W, H = em_nmf(V, W_init, H_init, n_iter=n_iter)

    # Component → heart/lung assignment
    assignments = []
    for k in range(K):
        cs_heart = cosine_sim(W[:, k], heart_template)
        cs_lung  = cosine_sim(W[:, k], lung_template)
        assignments.append("heart" if cs_heart > cs_lung else "lung")

    # Rebuild TF masks# Rebuild TF masks
    WH = W @ H + 1e-10

    # Component masks (F,K,T)
    comp_masks = (W[:, :, None] * H[None, :, :]) / (WH[:, None, :])

    # Component assignments
    heart_indices = [i for i, a in enumerate(assignments) if a == "heart"]
    lung_indices  = [i for i, a in enumerate(assignments) if a == "lung"]

    # Aggregate masks
    if len(heart_indices) > 0:
        heart_mask = np.sum(comp_masks[:, heart_indices, :], axis=1)
    else:
        heart_mask = np.zeros((W.shape[0], H.shape[1]))

    if len(lung_indices) > 0:
        lung_mask = np.sum(comp_masks[:, lung_indices, :], axis=1)
    else:
        lung_mask = np.zeros((W.shape[0], H.shape[1]))


    # Reconstruct signals
    heart_S = S * heart_mask
    lung_S  = S * lung_mask

    heart_est = istft(heart_S, hop_length=hop_length, length=len(mix_audio))
    lung_est  = istft(lung_S,  hop_length=hop_length, length=len(mix_audio))

    return heart_est.astype(np.float32), lung_est.astype(np.float32)
