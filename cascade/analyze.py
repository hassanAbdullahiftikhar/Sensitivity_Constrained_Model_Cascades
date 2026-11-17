"""Offline cascade analysis from saved logits (no GPU needed).

    python -m cascade.analyze --dataset pathmnist [--size 128] [--variant fp32|int8] [--calib val|target]

Threshold selection data ("selection set") vs evaluation data:
  --calib val     select on the source validation split, evaluate on the full test split. For PathMNIST the
                  validation split is from the training centre and the test split from another centre, so
                  this measures what happens when thresholds are tuned at the source site.
  --calib target  split the test data into a stratified calibration part (--calib-frac) used for selection and
                  an evaluation part, repeated over --n-splits random splits. This models a site that labels a
                  small local set before deployment; the LTT guarantee applies to this setting.
Temperatures are always fitted on the source validation split (a model property, not a routing choice).

For every seed found under runs/<dataset>_<size>/, pairs Stage-1 and Stage-2 of the same seed. Writes
results/<dataset>_<size>[_<variant>][_target]/:
  rows.csv              one row per seed x split x signal x alpha x rule x constraint (+ baselines, oracle)
  summary.csv           mean / std over seeds (and splits)
  signal_quality.csv    how well each signal ranks Stage-1 errors and Stage-1 missed positives (AUROC)
  calibration.csv       temperature and ECE before / after scaling
  fn_cost_ablation.csv  miss-risk signal at c_FN in {1, 2, 5, 10}
  curves_seed<k>.npz    evaluation-set sensitivity / accuracy vs routing rate per signal (first split)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from . import data
from .models import LARGE, SMALL
from .routing import cascade_outputs, classification_metrics, curves, select_k, threshold_for
from .signals import SIGNALS, ece, fit_temperature, miss_risk, probs
from .stats import bootstrap_ci, mcnemar_exact

ROOT = Path(__file__).resolve().parent.parent
ALPHAS = (0.95, 0.97, 0.99, 1.0)
RULES = ("plugin", "ltt")
CONSTRAINTS = ("sens", "sens+acc")
FN_COSTS = (1.0, 2.0, 5.0, 10.0)


@dataclass
class Part:
    """Stage-1 / Stage-2 probabilities (raw and temperature-scaled) and labels for one set of images."""
    small: np.ndarray
    small_ts: np.ndarray
    large: np.ndarray
    y: np.ndarray

    def take(self, idx: np.ndarray) -> "Part":
        return Part(self.small[idx], self.small_ts[idx], self.large[idx], self.y[idx])

    def p_small(self, ts: bool) -> np.ndarray:
        return self.small_ts if ts else self.small


def load_logits(run: Path, split: str, variant: str):
    name = f"logits_{split}.npz" if variant == "fp32" else f"logits_{split}_{variant}.npz"
    z = np.load(run / name)
    return z["logits"].astype(np.float64), z["labels"]


def evaluate_pair(spec, sel: Part, ev: Part, seed: int, split: int, n_boot: int, rng) -> tuple[list, list, dict, list]:
    pos, nc = spec.positive, spec.num_classes
    sp_sel, lp_sel = sel.small.argmax(1), sel.large.argmax(1)
    sp_ev, lp_ev = ev.small.argmax(1), ev.large.argmax(1)
    tag = {"seed": seed, "split": split}

    m_small = classification_metrics(ev.small, ev.y, pos, nc)
    m_large = classification_metrics(ev.large, ev.y, pos, nc)
    m_large_sel = classification_metrics(sel.large, sel.y, pos, nc)
    rows = [{**tag, "method": "always_small", "route_rate": 0.0, **m_small},
            {**tag, "method": "always_large", "route_rate": 1.0, **m_large}]
    oracle = (sp_ev != ev.y) & (lp_ev == ev.y)
    rows.append({**tag, "method": "oracle", "route_rate": float(oracle.mean()),
                 **classification_metrics(cascade_outputs(ev.small, ev.large, oracle), ev.y, pos, nc)})

    is_pos = np.isin(ev.y, pos)
    det_large = np.isin(lp_ev, pos) & is_pos
    quality, curve_dump = [], {}

    for sig, (fn, use_ts) in SIGNALS.items():
        u_sel, u_ev = fn(sel.p_small(use_ts), pos), fn(ev.p_small(use_ts), pos)
        c_sel = curves(u_sel, sp_sel, lp_sel, sel.y, pos)
        c_ev = curves(u_ev, sp_ev, lp_ev, ev.y, pos)
        curve_dump[f"{sig}_sens"], curve_dump[f"{sig}_acc"] = c_ev.sens, c_ev.acc

        small_wrong = sp_ev != ev.y
        missed = is_pos & ~np.isin(sp_ev, pos)
        quality.append({**tag, "signal": sig,
                        "auroc_small_error": float(roc_auc_score(small_wrong, u_ev)) if 0 < small_wrong.sum() < len(u_ev) else np.nan,
                        "auroc_small_miss_among_pos": float(roc_auc_score(missed[is_pos], u_ev[is_pos])) if 0 < missed.sum() < is_pos.sum() else np.nan})

        for alpha in ALPHAS:
            for constraint in CONSTRAINTS:
                acc_ref = m_large_sel["accuracy"] if constraint == "sens+acc" else None
                for rule in RULES:
                    k, feasible = select_k(c_sel, alpha, m_large_sel["sensitivity"], rule, acc_ref)
                    tau = threshold_for(c_sel, k)
                    routed = u_ev > tau
                    p_c = cascade_outputs(ev.small, ev.large, routed)
                    met = classification_metrics(p_c, ev.y, pos, nc)
                    pred_c = p_c.argmax(1)
                    det_c = np.isin(pred_c, pos) & is_pos

                    def sens_gap(idx):
                        return (det_c[idx].sum() - det_large[idx].sum()) / max(is_pos[idx].sum(), 1)

                    lo, hi = bootstrap_ci(sens_gap, len(ev.y), n_boot, seed) if n_boot else (np.nan, np.nan)
                    rr = float(routed.mean())
                    lost_rate = float((det_large & ~det_c).sum() / max(is_pos.sum(), 1))
                    rand_sens, rand_acc = [], []
                    for _ in range(100):
                        pr = np.where(rng.random(len(ev.y)) < rr, lp_ev, sp_ev)
                        rand_sens.append((np.isin(pr, pos) & is_pos).sum() / is_pos.sum())
                        rand_acc.append((pr == ev.y).mean())
                    rows.append({
                        **tag, "method": "cascade", "signal": sig, "alpha": alpha, "rule": rule,
                        "constraint": constraint, "feasible_on_sel": feasible, "threshold": tau, "route_rate": rr,
                        "sel_route_rate": k / len(sel.y),
                        "floor_met_test": met["sensitivity"] >= alpha * m_large["sensitivity"] - 1e-12,
                        "lost_det_rate": lost_rate,
                        "lost_det_ok": lost_rate <= (1 - alpha) * m_large_sel["sensitivity"] + 1e-12,
                        "sens_gap_vs_large": met["sensitivity"] - m_large["sensitivity"],
                        "sens_gap_ci_lo": lo, "sens_gap_ci_hi": hi,
                        "mcnemar_p_vs_large": mcnemar_exact(pred_c == ev.y, lp_ev == ev.y),
                        "random_sens_mean": float(np.mean(rand_sens)), "random_acc_mean": float(np.mean(rand_acc)),
                        **met,
                    })

    ablation = []
    for fn_cost in FN_COSTS:
        u_sel, u_ev = miss_risk(sel.small_ts, pos, fn_cost), miss_risk(ev.small_ts, pos, fn_cost)
        c_sel = curves(u_sel, sp_sel, lp_sel, sel.y, pos)
        for alpha in ALPHAS:
            for rule in RULES:
                k, _ = select_k(c_sel, alpha, m_large_sel["sensitivity"], rule)
                routed = u_ev > threshold_for(c_sel, k)
                pred_c = np.where(routed, lp_ev, sp_ev)
                ablation.append({**tag, "fn_cost": fn_cost, "alpha": alpha, "rule": rule,
                                 "route_rate": float(routed.mean()), "accuracy": float((pred_c == ev.y).mean()),
                                 "sensitivity": float((np.isin(pred_c, pos) & is_pos).sum() / is_pos.sum())})
    return rows, quality, curve_dump, ablation


def load_parts(base: Path, seed: int, variant: str):
    """Validation and test Parts for one numeric variant; temperatures fitted on that variant's validation logits."""
    ls_val, y_val = load_logits(base / SMALL / f"seed{seed}", "val", variant)
    ls_te, y_te = load_logits(base / SMALL / f"seed{seed}", "test", variant)
    ll_val, _ = load_logits(base / LARGE / f"seed{seed}", "val", variant)
    ll_te, _ = load_logits(base / LARGE / f"seed{seed}", "test", variant)
    t_s, t_l = fit_temperature(ls_val, y_val), fit_temperature(ll_val, y_val)
    calib_row = {"seed": seed, "T_small": t_s, "T_large": t_l,
                 "ece_small_raw": ece(probs(ls_te), y_te), "ece_small_ts": ece(probs(ls_te, t_s), y_te),
                 "ece_large_raw": ece(probs(ll_te), y_te), "ece_large_ts": ece(probs(ll_te, t_l), y_te)}
    val = Part(probs(ls_val), probs(ls_val, t_s), probs(ll_val), y_val)
    test = Part(probs(ls_te), probs(ls_te, t_s), probs(ll_te), y_te)
    return val, test, calib_row


def analyze_seed(spec, base: Path, seed: int, variant: str, select_variant: str, calib: str, n_splits: int,
                 calib_frac: float, n_boot: int):
    """Thresholds (and temperature) come from `select_variant` outputs and are applied to `variant` outputs;
    they differ only for the naive-deployment study (e.g. FP32-tuned thresholds on an INT8 model)."""
    val, test, calib_row = load_parts(base, seed, variant)
    if select_variant != variant:
        val_sel, test_sel, _ = load_parts(base, seed, select_variant)
        # the evaluated model's own temperature is replaced by the one the thresholds were tuned with
        t_sel = fit_temperature(*load_logits(base / SMALL / f"seed{seed}", "val", select_variant))
        ls_te, _ = load_logits(base / SMALL / f"seed{seed}", "test", variant)
        test = Part(test.small, probs(ls_te, t_sel), test.large, test.y)
    else:
        val_sel, test_sel = val, test
    if calib == "val":
        pairs = [(val_sel, test)]
    else:
        sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=calib_frac, random_state=1000 + seed)
        pairs = [(test_sel.take(c), test.take(e)) for c, e in sss.split(np.zeros(len(test.y)), test.y)]

    rng = np.random.default_rng(seed)
    rows, quality, ablation, first_curves = [], [], [], None
    for split, (sel, ev) in enumerate(pairs):
        r, q, cd, ab = evaluate_pair(spec, sel, ev, seed, split, n_boot, rng)
        rows += r
        quality += q
        ablation += ab
        first_curves = first_curves or cd
    return rows, quality, calib_row, first_curves, ablation


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(data.SPECS))
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--variant", default="fp32", help="fp32 (PyTorch logits) or e.g. int8 / ort_fp32")
    ap.add_argument("--select-variant", default=None,
                    help="variant whose outputs select thresholds (default: --variant). e.g. --variant int8 "
                         "--select-variant fp32 = FP32-tuned thresholds deployed on the INT8 model")
    ap.add_argument("--calib", default="val", choices=["val", "target"])
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--n-boot", type=int, default=None, help="default 1000 (val) / 0 (target: splits give the spread)")
    args = ap.parse_args(argv)
    n_boot = args.n_boot if args.n_boot is not None else (1000 if args.calib == "val" else 0)

    spec = data.SPECS[args.dataset]
    base = ROOT / "runs" / f"{args.dataset}_{args.size}"
    seeds = sorted(int(p.name[4:]) for p in (base / SMALL).glob("seed*")
                   if (base / LARGE / p.name / "logits_test.npz").exists())
    if not seeds:
        raise SystemExit(f"no paired seeds under {base}")
    select_variant = args.select_variant or args.variant
    name = f"{args.dataset}_{args.size}" + ("" if args.variant == "fp32" else f"_{args.variant}")
    name += "" if select_variant == args.variant else f"_sel{select_variant}"
    out = ROOT / "results" / (name + ("" if args.calib == "val" else "_target"))
    out.mkdir(parents=True, exist_ok=True)

    rows, quality, calib, ablation = [], [], [], []
    for seed in seeds:
        r, q, c, cd, ab = analyze_seed(spec, base, seed, args.variant, select_variant, args.calib, args.n_splits,
                                       args.calib_frac, n_boot)
        rows += r
        quality += q
        calib.append(c)
        ablation += ab
        np.savez(out / f"curves_seed{seed}.npz", **cd)
        print(f"seed {seed}: {len(r)} rows", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "rows.csv", index=False)
    pd.DataFrame(quality).to_csv(out / "signal_quality.csv", index=False)
    pd.DataFrame(calib).to_csv(out / "calibration.csv", index=False)
    pd.DataFrame(ablation).to_csv(out / "fn_cost_ablation.csv", index=False)

    keys = ["method", "signal", "alpha", "rule", "constraint"]
    cols = ["route_rate", "accuracy", "sensitivity", "specificity", "macro_recall", "auc_binary", "sens_gap_vs_large",
            "floor_met_test", "lost_det_rate", "lost_det_ok", "random_sens_mean", "random_acc_mean"]
    g = df.copy()
    g[keys] = g[keys].astype(object).fillna("-").astype(str)
    g[["floor_met_test", "lost_det_ok"]] = g[["floor_met_test", "lost_det_ok"]].astype(float)
    summ = g.groupby(keys, sort=False)[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(out / "summary.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
