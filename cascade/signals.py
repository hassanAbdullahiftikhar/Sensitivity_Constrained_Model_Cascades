"""Routing signals computed from Stage-1 outputs, temperature scaling and calibration error.

Every signal returns an "escalation score" u: higher u means route the image to Stage 2.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def probs(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return softmax(logits / temperature, axis=1)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): minimise validation NLL over a single scalar T."""

    def nll(log_t: float) -> float:
        return -log_softmax(logits / np.exp(log_t), axis=1)[np.arange(len(labels)), labels].mean()

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def ece(p: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf, pred = p.max(1), p.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            err += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(err)


def msp(p: np.ndarray, positive) -> np.ndarray:
    return 1.0 - p.max(1)


def entropy(p: np.ndarray, positive) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(1)


def margin(p: np.ndarray, positive) -> np.ndarray:
    top2 = np.sort(p, axis=1)[:, -2:]
    return 1.0 - (top2[:, 1] - top2[:, 0])


def miss_risk(p: np.ndarray, positive, fn_cost: float = 5.0) -> np.ndarray:
    """Bayes risk of accepting the Stage-1 decision when a missed positive costs `fn_cost` and any other
    error costs 1. Reduces to MSP when fn_cost == 1; for predicted negatives it adds weight to the
    probability mass on positive classes, which is the mass that can turn into a false negative."""
    pred_neg = ~np.isin(p.argmax(1), positive)
    p_pos = p[:, list(positive)].sum(1)
    return (1.0 - p.max(1)) + (fn_cost - 1.0) * p_pos * pred_neg


# name -> (function, uses temperature-scaled probabilities)
SIGNALS = {
    "msp": (msp, False),
    "entropy": (entropy, False),
    "margin": (margin, False),
    "ts_msp": (msp, True),
    "ts_entropy": (entropy, True),
    "miss_risk": (miss_risk, True),
}
