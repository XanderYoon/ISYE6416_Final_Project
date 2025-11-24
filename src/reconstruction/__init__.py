# src/separation/__init__.py

"""
Separation package: utilities for template-based heart / lung source separation.

Main public API (import from here in notebooks):

- from separation.templates import build_heart_lung_templates
- from separation.mask_separation import (
      separate_mixture_from_audio,
      evaluate_single_mixture,
      evaluate_over_dataframe,
  )
"""
from .templates import build_heart_lung_templates
from .stft_utils import (
    compute_stft,
    istft,
    magnitude_spectrogram,
    db_spectrogram
)
from .nmf_kgrid import (
    cosine_sim,
    em_nmf_k_components
)
from .mask_separation import (
    SeparationResult,
    compute_soft_masks_from_templates,
    separate_mixture_from_audio,
    _align_signals,
    waveform_mse,
    waveform_corr,
    waveform_r2,
    spectrogram_mse,
    evaluate_single_mixture,
    evaluate_over_dataframe
)
from .hmm_smoothing import hmm_smooth_posteriors
from .em_nmf import (
    em_nmf,
    EMNMFResult,
    em_nmf_separate,
    em_nmf_hmm_separate,
    _trim,
    mse,
    corr,
    spec_mse,
    r2
)