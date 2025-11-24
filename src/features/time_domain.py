from typing import Dict

import librosa
import numpy as np


def compute_time_domain_features(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """
    Compute simple time-domain features: RMS, ZCR, duration.
    """
    rms = float(np.sqrt(np.mean(audio ** 2)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    duration = float(len(audio) / sample_rate)

    return {
        "rms": rms,
        "zcr": zcr,
        "duration": duration,
    }
