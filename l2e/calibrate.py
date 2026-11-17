"""Temperature scaling and calibration error.

Temperature scaling divides every logit by one number T, fit on the validation set to minimize log
loss. It changes how confident the scores are, not their ranking, and at threshold 0.5 not the verdict
either. Every monitor gets it, so the confidence-based deferral baselines are fair.
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def nll(logits, y, T):
    p = np.clip(sigmoid(np.asarray(logits) / T), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(logits, y):
    """Grid search over log T, then a finer grid around the best value."""
    logits, y = np.asarray(logits, float), np.asarray(y, float)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 300))
    best = min(grid, key=lambda T: nll(logits, y, T))
    fine = np.linspace(best * 0.9, best * 1.1, 200)
    return float(min(fine, key=lambda T: nll(logits, y, T)))


def ece(p, y, bins=15):
    """Expected calibration error with equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))
