"""Paired statistics on a shared test set: bootstrap CIs, exact McNemar, Clopper-Pearson bounds."""
from __future__ import annotations

import numpy as np
from scipy.stats import beta, binomtest


def clopper_pearson_lower(successes: int, n: int, delta: float) -> float:
    """One-sided (1 - delta) lower confidence bound for a binomial proportion."""
    return 0.0 if successes == 0 else float(beta.ppf(delta, successes, n - successes + 1))


def clopper_pearson_upper(successes: int, n: int, delta: float) -> float:
    """One-sided (1 - delta) upper confidence bound for a binomial proportion."""
    return 1.0 if successes >= n else float(beta.ppf(1 - delta, successes + 1, n - successes))


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Two-sided exact McNemar p-value on paired per-sample correctness."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    return 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)


def bootstrap_ci(stat, n: int, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of stat(idx) over resamples idx of range(n). stat must accept an index array."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    a = (1 - level) / 2
    return float(np.nanquantile(vals, a)), float(np.nanquantile(vals, 1 - a))
