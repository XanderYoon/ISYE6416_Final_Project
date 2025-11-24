from typing import Dict

import librosa
import numpy as np


def compute_mfcc_summary(audio: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> Dict[str, np.ndarray]:
    """
    Compute MFCC features and summarize with mean and std per coefficient.

    Returns
    -------
    {
        "mfcc_mean": np.ndarray, shape (n_mfcc,),
        "mfcc_std": np.ndarray, shape (n_mfcc,),
    }
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return {
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
    }
