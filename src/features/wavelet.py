from typing import Any

import numpy as np
import pywt


def compute_wavelet_energy(
    audio: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> np.ndarray:
    """
    Compute normalized wavelet band energies.
    """
    coeffs = pywt.wavedec(audio, wavelet, level=level)
    energies = np.array([np.sum(c ** 2) for c in coeffs], dtype=float)
    total_energy = float(np.sum(energies)) or 1.0
    return energies / total_energy
