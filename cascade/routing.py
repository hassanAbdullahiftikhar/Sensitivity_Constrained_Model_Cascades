"""Two-stage cascade: threshold selection under a sensitivity floor and evaluation of the cascade.

Sensitivity is binary: an image is "positive" if its true class is in spec.positive, and it is detected if
the predicted class is also in spec.positive. The floor is relative to the always-large model:
    sens(cascade) >= alpha * sens(large), with sens(large) measured on the selection (validation) split.

Two selection rules:
  plugin  - smallest routing rate whose empirical selection-set sensitivity meets the floor.
  ltt     - paired Learn-then-Test (Angelopoulos et al.). The floor is rewritten as a bound on the rate of
            *lost detections*: positives that Stage 2 detects but the cascade misses (accepted at Stage 1 and
            missed there), as a fraction of positives, must be <= eps = (1 - alpha) * sens(large). Since
            sens(cascade) >= sens(large) - lost rate, this bounds the sensitivity gap. The lost rate is zero when
            everything is routed and grows monotonically as fewer images are routed, so fixed-sequence testing
            walks from route-everything towards route-nothing and stops at the first threshold whose one-sided
            Clopper-Pearson upper bound exceeds eps. With probability >= 1 - delta (exchangeable data) the chosen
            threshold keeps the population lost-detection rate <= eps. alpha = 1 (eps = 0) cannot be certified
            with finite data and therefore routes everything.
An optional accuracy floor is handled the same way (lost-correct rate <= (1 - alpha) * acc(large)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import clopper_pearson_upper


@dataclass
class Curves:
    u_sorted: np.ndarray   # escalation scores, descending
    acc: np.ndarray        # cascade accuracy when the top-k scores are routed, k = 0..n
    sens: np.ndarray       # cascade binary sensitivity, k = 0..n
    lost_det: np.ndarray   # count of positives Stage 2 detects but the cascade misses, k = 0..n
    lost_acc: np.ndarray   # count of images Stage 2 gets right but the cascade gets wrong, k = 0..n
    n_pos: int
    valid_k: np.ndarray    # k values realisable by a strict threshold (no ties split)


def curves(u: np.ndarray, small_pred: np.ndarray, large_pred: np.ndarray, y: np.ndarray, positive) -> Curves:
    order = np.argsort(-u, kind="stable")
    is_pos = np.isin(y, positive)[order]
    sc, lc = (small_pred == y)[order], (large_pred == y)[order]
    sd = (np.isin(small_pred, positive)[order] & is_pos)
    ld = (np.isin(large_pred, positive)[order] & is_pos)
    cum = lambda a: np.concatenate([[0], np.cumsum(a)])
    n, n_pos = len(y), int(is_pos.sum())
    acc = (cum(lc) + sc.sum() - cum(sc)) / n
    sens = (cum(ld) + sd.sum() - cum(sd)) / max(n_pos, 1)
    lost_d, lost_a = ld & ~sd, lc & ~sc  # harm from accepting Stage 1 instead of routing
    lost_det = lost_d.sum() - cum(lost_d)
    lost_acc = lost_a.sum() - cum(lost_a)
    us = u[order]
    valid = np.concatenate([[0], np.flatnonzero(us[:-1] > us[1:]) + 1, [n]])
    return Curves(us, acc, sens, lost_det, lost_acc, n_pos, valid)


def threshold_for(c: Curves, k: int) -> float:
    """Route iff u > threshold; routes exactly the top-k on the data the curves were built from."""
    if k == 0:
        return float(np.inf)
    if k == len(c.u_sorted):
        return float(-np.inf)
    return float((c.u_sorted[k - 1] + c.u_sorted[k]) / 2)


def select_k(c: Curves, alpha: float, sens_large: float, rule: str = "plugin", acc_large: float | None = None,
             delta: float = 0.05) -> tuple[int, bool]:
    """Fewest routed images meeting the floor(s). Returns (k, feasible); infeasible -> route everything.
    sens_large / acc_large are Stage 2's metrics on the same selection set; acc_large=None drops the accuracy floor."""
    n = len(c.u_sorted)
    if rule == "plugin":
        ok = c.sens[c.valid_k] >= alpha * sens_large - 1e-12
        if acc_large is not None:
            ok &= c.acc[c.valid_k] >= alpha * acc_large - 1e-12
        ks = c.valid_k[ok]
        return (int(ks.min()), True) if len(ks) else (n, False)
    if rule == "ltt":
        d = delta / 2 if acc_large is not None else delta
        eps_d = (1 - alpha) * sens_large
        eps_a = (1 - alpha) * acc_large if acc_large is not None else None
        chosen = None
        for k in c.valid_k[::-1]:
            ok = clopper_pearson_upper(int(c.lost_det[k]), c.n_pos, d) <= eps_d
            if eps_a is not None:
                ok &= clopper_pearson_upper(int(c.lost_acc[k]), n, d) <= eps_a
            if not ok:
                break
            chosen = int(k)
        return (chosen, True) if chosen is not None else (n, False)
    raise ValueError(rule)


def binary_metrics(pred: np.ndarray, y: np.ndarray, positive) -> dict:
    is_pos, called = np.isin(y, positive), np.isin(pred, positive)
    return {
        "sensitivity": float((called & is_pos).sum() / max(is_pos.sum(), 1)),
        "specificity": float((~called & ~is_pos).sum() / max((~is_pos).sum(), 1)),
    }


def classification_metrics(p: np.ndarray, y: np.ndarray, positive, num_classes: int) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score

    from .signals import ece

    pred = p.argmax(1)
    recalls = [float((pred[y == c] == c).mean()) if (y == c).any() else np.nan for c in range(num_classes)]
    is_pos = np.isin(y, positive)
    return {
        "accuracy": float((pred == y).mean()),
        "macro_recall": float(np.nanmean(recalls)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        **binary_metrics(pred, y, positive),
        "auc_binary": float(roc_auc_score(is_pos, p[:, list(positive)].sum(1))) if 0 < is_pos.sum() < len(y) else np.nan,
        "ece": ece(p, y),
        **{f"recall_c{c}": r for c, r in enumerate(recalls)},
    }


def cascade_outputs(p_small: np.ndarray, p_large: np.ndarray, routed: np.ndarray) -> np.ndarray:
    return np.where(routed[:, None], p_large, p_small)
