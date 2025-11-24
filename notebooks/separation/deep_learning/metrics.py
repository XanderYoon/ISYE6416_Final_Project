import numpy as np
from scipy.stats import pearsonr


def mse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    L = min(len(a), len(b))
    return np.mean((a[:L] - b[:L])**2)


def corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    L = min(len(a), len(b))
    if L < 2:
        return 0.0
    r, _ = pearsonr(a[:L], b[:L])
    return float(r)


def r2(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    L = min(len(a), len(b))
    if L < 2:
        return 0.0
    ss_res = np.sum((a[:L] - b[:L])**2)
    ss_tot = np.sum((a[:L] - np.mean(a[:L]))**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
