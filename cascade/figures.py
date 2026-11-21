"""Paper figures (PDF + PNG) from analysis and benchmark results.

    python -m cascade.figures

Writes to paper/figures/. Colours follow a CVD-validated categorical order (blue, orange, aqua); every series
also carries a distinct line style / marker and a direct label so identity never rests on colour alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {  # signal -> (label, colour, linestyle, marker)
    "msp": ("MSP", "#2a78d6", "-", "o"),
    "entropy": ("Entropy", "#eb6834", "--", "s"),
    "miss_risk": ("Miss-risk", "#1baf7a", "-.", "^"),
}
DATASETS = {"pathmnist": "PathMNIST (histopathology)", "ham10000": "HAM10000 (dermoscopy)"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf/.png")


def sensitivity_vs_escalation(datasets: list[str], max_rate: float = 0.3) -> None:
    """Full-test-set sensitivity vs. share of images escalated, averaged over seeds (curves from --calib val,
    which are computed on the whole test split and do not depend on threshold selection)."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.5 * len(datasets), 2.7), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d = ROOT / "results" / f"{ds}_128"
        files = sorted(d.glob("curves_seed*.npz"))
        rows = pd.read_csv(d / "rows.csv")
        s_small = rows[rows.method == "always_small"].sensitivity.mean() * 100
        s_large = rows[rows.method == "always_large"].sensitivity.mean() * 100
        ax.plot([0, 100], [s_small, s_large], color=MUTED, lw=1.2, ls=":", zorder=1)
        ax.axhline(s_large, color=INK2, lw=0.8, zorder=1)
        ax.text(0.98, s_large, "always ResNet-18", transform=ax.get_yaxis_transform(), color=INK2,
                ha="right", va="bottom", fontsize=7)
        ax.text(0.98, s_small + (s_large - s_small) * max_rate, "random escalation",
                transform=ax.get_yaxis_transform(), color=MUTED, ha="right", va="top", fontsize=7)
        for sig, (label, col, ls, mk) in SERIES.items():
            y = np.mean([np.load(f)[f"{sig}_sens"] for f in files], axis=0) * 100
            x = np.linspace(0, 100, len(y))
            keep = x <= max_rate * 100
            ax.plot(x[keep], y[keep], color=col, lw=1.6, ls=ls, zorder=3, label=label, marker=mk,
                    markevery=max(1, keep.sum() // 5), markersize=4, markeredgecolor="#fcfcfb", markeredgewidth=0.8)
        ax.set_xlim(0, max_rate * 100)
        ax.set_title(f"{DATASETS[ds]}, mean of {len(files)} seeds", color=INK, fontsize=8.5, loc="left")
        ax.set_xlabel("Images escalated to ResNet-18 (%)")
        ax.set_ylabel("Cascade sensitivity (%)")
        ax.legend(loc="center right", fontsize=7, bbox_to_anchor=(1.0, 0.45))
    save(fig, "sensitivity_vs_escalation")


def budget_table(datasets: list[str], budgets=(0.02, 0.05, 0.10, 0.20)) -> None:
    """Test sensitivity / accuracy at fixed escalation budgets (no threshold selection involved)."""
    from .signals import SIGNALS

    recs = []
    for ds in datasets:
        for f in sorted((ROOT / "results" / f"{ds}_128").glob("curves_seed*.npz")):
            z = np.load(f)
            for sig in SIGNALS:
                n = len(z[f"{sig}_sens"]) - 1
                for b in budgets:
                    k = int(round(b * n))
                    recs.append({"dataset": ds, "signal": sig, "budget_pct": int(b * 100),
                                 "sensitivity": z[f"{sig}_sens"][k] * 100, "accuracy": z[f"{sig}_acc"][k] * 100})
    df = pd.DataFrame(recs)
    g = df.groupby(["dataset", "budget_pct", "signal"], sort=False)
    m, s = g.mean(numeric_only=True), g.std(numeric_only=True)
    out = (m.round(2).astype(str) + " ± " + s.round(2).astype(str)).reset_index()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "budget_table.md").write_text(out.to_markdown(index=False), encoding="utf-8")
    print(out.to_string(index=False))


def quantization_table(datasets: list[str], calib: str = "target") -> None:
    """FP32 vs INT8 (own thresholds) vs INT8 with FP32-tuned thresholds (naive deployment)."""
    from .models import SMALL

    suffix = "_target" if calib == "target" else ""
    recs = []
    for ds in datasets:
        for setting, name in (("FP32", ""), ("INT8 naive", "_int8naive"),
                              ("INT8 naive, FP32 thresholds", "_int8naive_selfp32"),
                              ("INT8 mixed", "_int8"), ("INT8 mixed, FP32 thresholds", "_int8_selfp32")):
            d = ROOT / "results" / f"{ds}_128{name}{suffix}"
            if not (d / "rows.csv").exists():
                continue
            r = pd.read_csv(d / "rows.csv")
            r[["floor_met_test", "lost_det_ok"]] = r[["floor_met_test", "lost_det_ok"]].astype(float)
            for m in ("always_small", "always_large"):
                b = r[r.method == m]
                recs.append({"dataset": ds, "setting": setting, "method": m, "route_rate": b.route_rate.mean(),
                             "sensitivity": b.sensitivity.mean(), "accuracy": b.accuracy.mean()})
            c = r[(r.method == "cascade") & (r.constraint == "sens") & r.signal.isin(["msp", "miss_risk"])
                  & r.alpha.isin([0.97, 0.99])]
            for (rule, alpha, sig), g in c.groupby(["rule", "alpha", "signal"]):
                recs.append({"dataset": ds, "setting": setting, "method": f"{sig} {rule} a={alpha}",
                             "route_rate": g.route_rate.mean(), "sensitivity": g.sensitivity.mean(),
                             "accuracy": g.accuracy.mean(), "floor_met": g.floor_met_test.mean(),
                             "lost_ok": g.lost_det_ok.mean()})
        # prediction agreement between FP32 (PyTorch) and INT8 (ORT) on test
        for seed_dir in sorted((ROOT / "runs" / f"{ds}_128" / SMALL).glob("seed*")):
            for arch_dir in (seed_dir, ROOT / "runs" / f"{ds}_128" / "resnet18" / seed_dir.name):
                a = np.load(arch_dir / "logits_test.npz")["logits"].argmax(1)
                for v in ("int8naive", "int8"):
                    q = arch_dir / f"logits_test_{v}.npz"
                    if q.exists():
                        recs.append({"dataset": ds, "setting": f"FP32-{v} agreement",
                                     "method": f"{arch_dir.parent.name} {seed_dir.name}",
                                     "accuracy": float((a == np.load(q)["logits"].argmax(1)).mean())})
    df = pd.DataFrame(recs)
    num = [c for c in df.columns if c not in ("dataset", "setting", "method")]
    df[num] = (df[num] * 100).round(2)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"quantization_table{suffix}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    print(df.to_string(index=False))


DEVICE_STYLE = {  # device tag -> (label, colour); fixed order so a device keeps its colour across figures
    "ryzen7700": ("Ryzen 7 7700 (x86 ref.)", "#2a78d6"),
    "oracle_a1": ("Ampere A1 (Neoverse-N1)", "#eb6834"),
    "android": ("Snapdragon 662, Kryo 260 Gold (A73-class)", "#1baf7a"),
    "android_little": ("Snapdragon 662, Kryo 260 Silver (A53-class)", "#eda100"),
}


def latency_vs_flops(bench_dirs: list[Path]) -> None:
    """Measured speed-up of the cascade over always-ResNet-18 vs. the FLOP-based prediction.
    FLOP prediction: F2 / (F1 + r * F2); measured: t(Stage 2 only) / t(cascade), both at batch 1."""
    recs = []
    for f in [p for d in bench_dirs for p in d.glob("*.json")]:
        r = json.loads(f.read_text())
        info = json.loads((ROOT / "deploy" / r["deploy"] / "model_info.json").read_text())
        f1, f2, rr = info["small"]["flops"], info["large"]["flops"], r["cascade"]["route_rate"]
        recs.append({"device": r["tag"], "deploy": r["deploy"], "precision": r["precision"], "threads": r["threads"],
                     "spin": r.get("spin", True), "signal": r["config"]["signal"], "alpha": r["config"]["alpha"], "route_rate": rr,
                     "stage1_ms": r["stage1_only"]["mean_ms"], "stage2_ms": r["stage2_only"]["mean_ms"],
                     "cascade_ms": r["cascade"]["mean_ms"], "cascade_p95_ms": r["cascade"]["p95_ms"],
                     "cascade_ci95_ms": r["cascade"]["ci95_half_ms"],
                     "latency_ratio_s2_s1": r["stage2_only"]["mean_ms"] / r["stage1_only"]["mean_ms"],
                     "flop_ratio_s2_s1": f2 / f1,
                     "speedup_measured": r["stage2_only"]["mean_ms"] / r["cascade"]["mean_ms"],
                     "speedup_flops": f2 / (f1 + rr * f2)})
    if not recs:
        print("no benchmark results yet")
        return
    df = pd.DataFrame(recs).sort_values(["device", "deploy", "precision", "threads", "spin", "route_rate"])
    df["model_err_pct"] = 100 * (df.cascade_ms / (df.stage1_ms + df.route_rate * df.stage2_ms) - 1)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "bench_summary.csv", index=False)
    print(df.round(3).to_string(index=False))

    # One panel per precision; x = escalation rate, y = speed-up; dashed line = FLOP prediction.
    from matplotlib.ticker import FixedLocator, NullLocator

    sub = df[df.deploy.str.startswith("pathmnist") & (df.threads == 1) & df.spin & df.precision.isin(["fp32", "int8"])]
    precisions = [p for p in ("fp32", "int8") if (sub.precision == p).any()]
    if not precisions:
        return
    fig, axes = plt.subplots(1, len(precisions), figsize=(3.5 * len(precisions), 2.8), squeeze=False, sharey=True)
    f1f2 = sub.flop_ratio_s2_s1.iloc[0]
    xs = np.linspace(0.0, 0.3, 200)
    for ax, prec in zip(axes[0], precisions):
        ax.plot(xs * 100, 1 / (1 / f1f2 + xs), color=MUTED, lw=1.2, ls="--", label="FLOP prediction")
        for dev, (label, col) in DEVICE_STYLE.items():
            d = sub[(sub.device == dev) & (sub.precision == prec)]
            if not len(d):
                continue
            s1, s2 = d.stage1_ms.mean(), d.stage2_ms.mean()
            ax.plot(xs * 100, s2 / (s1 + xs * s2), color=col, lw=1.0, alpha=0.6, zorder=2)
            mk = {"android": "s", "android_little": "^", "oracle_a1": "D"}.get(dev, "o")
            ax.plot(d.route_rate * 100, d.speedup_measured, ls="none", marker=mk, markersize=6, color=col,
                    markeredgecolor="#fcfcfb", markeredgewidth=1, zorder=3, label=label)
        ax.axhline(1, color=AXIS, lw=0.8)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 30]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.set_xlim(0, 30)
        ax.set_title(f"PathMNIST, {prec.upper()}, 1 thread", color=INK, fontsize=8.5, loc="left")
        ax.set_xlabel("Images escalated to ResNet-18 (%)")
    axes[0][0].set_ylabel("Speed-up over always-ResNet-18 (×)")
    # one shared legend below both panels so it never covers data; the thin lines are explained in the caption
    from matplotlib.lines import Line2D

    handles, labels = axes[0][0].get_legend_handles_labels()
    for h, l in zip(*axes[0][-1].get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)
    handles.append(Line2D([], [], color=MUTED, lw=1.0, alpha=0.6))
    labels.append("Measured stage latencies: $t_L/(t_S + r\\,t_L)$")
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.4, handletextpad=0.5)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.31, wspace=0.08)
    save(fig, "speedup_vs_escalation")


if __name__ == "__main__":
    have = [ds for ds in DATASETS if (ROOT / "results" / f"{ds}_128" / "curves_seed0.npz").exists()]
    sensitivity_vs_escalation(have)
    extra = [ds for ds in ("ham10000bal",) if (ROOT / "results" / f"{ds}_128" / "curves_seed0.npz").exists()]
    budget_table(have + extra)  # class-balanced HAM10000 is a robustness check: table only, not the main figure
    quantization_table(have, "target")
    quantization_table(have, "val")
    latency_vs_flops([p for p in (ROOT / "results" / "bench").glob("*") if p.is_dir()])
