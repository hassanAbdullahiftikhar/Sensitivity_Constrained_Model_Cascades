"""Markdown tables (mean ± std over seeds) from an analysis directory.

    python -m cascade.report results/pathmnist_128 [results/pathmnist_128_int8 ...]

Writes report.md into each directory and prints it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def pm(mean: pd.Series, std: pd.Series, pct: bool = True) -> pd.Series:
    k = 100 if pct else 1
    return [f"{m * k:.2f} ± {s * k:.2f}" if pd.notna(s) else f"{m * k:.2f}" for m, s in zip(mean, std)]


def table(df: pd.DataFrame, by: list[str], cols: list[str]) -> str:
    g = df.groupby(by, sort=False)[cols]
    m, s = g.mean(), g.std()
    out = pd.DataFrame(index=m.index)
    for c in cols:
        out[c] = pm(m[c], s[c])
    return out.reset_index().to_markdown(index=False)


def report(d: Path) -> str:
    rows = pd.read_csv(d / "rows.csv")
    rows[["floor_met_test", "lost_det_ok"]] = rows[["floor_met_test", "lost_det_ok"]].astype(float)
    n_seeds, n_splits = rows.seed.nunique(), rows.split.nunique()
    parts = [f"# {d.name} ({n_seeds} seeds x {n_splits} calibration splits; values in %, mean ± std)\n"]

    base = rows[rows.method.isin(["always_small", "always_large", "oracle"])]
    parts += ["## Baselines (test)\n",
              table(base, ["method"], ["route_rate", "accuracy", "sensitivity", "specificity", "auc_binary", "macro_recall"])]

    casc = rows[rows.method == "cascade"]
    main_cols = ["route_rate", "sensitivity", "accuracy", "floor_met_test", "lost_det_rate", "lost_det_ok",
                 "random_sens_mean"]
    for constraint in casc.constraint.unique():
        for alpha in sorted(casc.alpha.unique()):
            sub = casc[(casc.alpha == alpha) & (casc.constraint == constraint)]
            parts += [f"\n## Cascade, floor α = {alpha}, constraint = {constraint} (test)\n",
                      "`floor_met_test` = share of runs meeting the floor on evaluation data; `lost_det_rate` = positives "
                      "Stage 2 detects but the cascade misses; `lost_det_ok` = share of runs with lost rate <= (1-α)·sens(large); "
                      "`random_sens_mean` = random routing at the same rate.\n",
                      table(sub, ["rule", "signal"], main_cols)]

    q = pd.read_csv(d / "signal_quality.csv")
    parts += ["\n## Signal quality (test AUROC)\n",
              table(q, ["signal"], ["auroc_small_error", "auroc_small_miss_among_pos"])]

    c = pd.read_csv(d / "calibration.csv")
    parts += ["\n## Calibration (test ECE, %; T fitted on val)\n",
              c.agg(["mean", "std"]).T.map(lambda v: f"{v:.4f}").to_markdown()]

    ab = d / "fn_cost_ablation.csv"
    if ab.exists():
        a = pd.read_csv(ab)
        parts += ["\n## Miss-risk c_FN ablation (test)\n",
                  table(a, ["alpha", "rule", "fn_cost"], ["route_rate", "sensitivity", "accuracy"])]
    text = "\n".join(parts) + "\n"
    (d / "report.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for arg in sys.argv[1:]:
        print(report(Path(arg)))
