# src/separation/hmm_smoothing.py

import numpy as np


def hmm_smooth_posteriors(
    heart_fraction: np.ndarray,
    p_stay: float = 0.99,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    2-state HMM smoothing over time.

    States:
        0 = heart-dominant
        1 = lung-dominant

    Observations:
        heart_fraction[t] ~ "noisy" p(z_t = heart | NMF)

    We treat emissions as:
        p(o_t | heart) ∝ heart_fraction[t]
        p(o_t | lung)  ∝ 1 - heart_fraction[t]

    Then run forward-backward with high self-transition probability.

    Parameters
    ----------
    heart_fraction : np.ndarray, shape (T,)
        Per-frame heart "probability" from NMF (0..1).
    p_stay : float
        Probability of staying in same state.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    gamma_heart : np.ndarray, shape (T,)
        Smoothed posterior p(z_t = heart | observations).
    """
    T = heart_fraction.shape[0]
    K = 2

    # Clamp to (eps, 1-eps)
    r = np.clip(heart_fraction, eps, 1.0 - eps)

    # Transition matrix
    p_switch = 1.0 - p_stay
    A = np.array([[p_stay, p_switch],
                  [p_switch, p_stay]], dtype=float)

    # Initial distribution
    pi = np.array([0.5, 0.5], dtype=float)

    # Emission probabilities e[t, s]
    e = np.zeros((T, K), dtype=float)
    e[:, 0] = r          # heart-dominant
    e[:, 1] = 1.0 - r    # lung-dominant

    # Forward with scaling
    alpha = np.zeros((T, K), dtype=float)
    c = np.zeros(T, dtype=float)

    alpha[0] = pi * e[0]
    c[0] = np.sum(alpha[0]) + eps
    alpha[0] /= c[0]

    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * e[t]
        c[t] = np.sum(alpha[t]) + eps
        alpha[t] /= c[t]

    # Backward with scaling
    beta = np.zeros((T, K), dtype=float)
    beta[-1] = 1.0 / (c[-1] + eps)

    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (e[t + 1] * beta[t + 1])) / (c[t] + eps)

    # Posterior gamma_t ∝ alpha_t * beta_t
    gamma = alpha * beta
    gamma /= np.sum(gamma, axis=1, keepdims=True) + eps

    # Return posterior of heart-dominant state
    gamma_heart = gamma[:, 0]
    return gamma_heart
