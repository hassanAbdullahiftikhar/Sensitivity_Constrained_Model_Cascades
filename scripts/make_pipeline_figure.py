"""Overview figure (Fig. 1), built from real test images and the real seed-0 model outputs.

    .venv\\Scripts\\python.exe scripts\\make_pipeline_figure.py   ->  paper/figures/pipeline.pdf / .png

One lane per test image: what the small model says, a bar for its miss-risk score u (Eq. 2) relative to the
certified threshold tau of the deployed config, the routing decision and, if escalated, what ResNet-18 says.
A single dashed tau line runs from the calibration step through every risk bar. Values are printed for checking.
Colour scheme: grey = structure, green = fast path / accepted, orange = risk / escalated, blue = calibration.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"

INK, INK2, MUTED, LINE = "#1f1f1d", "#5a5955", "#8f8d87", "#c9c7bf"
GREEN, ORANGE, BLUE, RED = "#16a06f", "#e6682e", "#2a6fd6", "#d64545"
NEUTRAL_FILL, GREEN_FILL, ORANGE_FILL, BLUE_FILL, TRACK = "#f4f3ef", "#e8f6f0", "#fdf0e8", "#edf3fc", "#e4e2db"

DATASETS = {
    "pathmnist": dict(name="Histopathology", data="data/medmnist/pathmnist_128.npz", positive=[8],
                      classes="ADI BACK DEB LYM MUC MUS NORM STR TUM".split(), cancer="tumour"),
    "ham10000": dict(name="Dermoscopy", data="data/ham10000/ham10000_128_split0.npz", positive=[0, 1, 4],
                     classes="akiec bcc bkl df mel nv vasc".split(), cancer="cancer"),
}
READABLE = {
    "ADI": "Fat tissue", "BACK": "Background", "DEB": "Debris", "LYM": "Lymphocytes", "MUC": "Mucus",
    "MUS": "Muscle", "NORM": "Normal tissue", "STR": "Stroma", "TUM": "Tumour",
    "akiec": "Actinic keratosis", "bcc": "Basal cell carcinoma", "bkl": "Benign keratosis",
    "df": "Dermatofibroma", "mel": "Melanoma", "nv": "Benign mole", "vasc": "Vascular lesion",
}
# (dataset, test index): accepted negative, accepted positive, two missed positives rescued by escalation
LANES = [("pathmnist", 7), ("pathmnist", 0), ("pathmnist", 652), ("ham10000", 213)]
TAU_FRAC = 0.5  # position of tau along every risk bar


def softmax(z: np.ndarray, t: float) -> np.ndarray:
    z = z / t
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def lane_info(ds: str, i: int) -> dict:
    spec = DATASETS[ds]
    cfg = json.loads((ROOT / "deploy" / f"{ds}_128_seed0" / "cascade_fp32_miss_risk_a0.97_ltt.json").read_text())
    s = np.load(ROOT / "runs" / f"{ds}_128" / "mobilenet_v3_small" / "seed0" / "logits_test.npz")
    big = np.load(ROOT / "runs" / f"{ds}_128" / "resnet18" / "seed0" / "logits_test.npz")
    z = np.load(ROOT / spec["data"])
    p = softmax(s["logits"][i], cfg["temperature"])
    pl = softmax(big["logits"][i], 1.0)
    yh, pos = int(p.argmax()), spec["positive"]
    p_pos = float(p[pos].sum())
    u = (1 - p.max()) + 4.0 * p_pos * (yh not in pos)  # c_FN = 5
    esc = bool(u > cfg["threshold"])
    y = int(np.asarray(z["test_labels"]).reshape(-1)[i])
    names = [READABLE[c] for c in spec["classes"]]
    return dict(img=z["test_images"][i], ds=spec["name"], names=names, y=y, yh=yh, p_pos=p_pos, u=float(u),
                tau=float(cfg["threshold"]), esc=esc, yl=int(pl.argmax()), final=int(pl.argmax()) if esc else yh,
                cancer=spec["cancer"], pred_pos=yh in pos, true_pos=y in pos)


# ---------------------------------------------------------------------------------------------------- drawing
def rbox(ax, x0, y0, w, h, fc, ec=LINE, lw=0.9, ls="-", r=0.06, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z))


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.0, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                arrowprops=dict(arrowstyle="-|>,head_length=0.32,head_width=0.16", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0))


def icon_bolt(ax, cx, cy, s=0.075, color=GREEN):
    pts = np.array([[0.1, 1], [-0.45, -0.05], [0.0, -0.05], [-0.15, -1], [0.45, 0.15], [0.0, 0.15], [0.25, 1]])
    ax.add_patch(Polygon(pts * s + [cx, cy], closed=True, fc=color, ec="none", zorder=6))


def icon_magnifier(ax, cx, cy, s=0.075, color=ORANGE):
    ax.add_patch(Circle((cx - 0.25 * s, cy + 0.25 * s), 0.55 * s, fc="white", ec=color, lw=1.3, zorder=6))
    ax.plot([cx + 0.15 * s, cx + 0.8 * s], [cy - 0.15 * s, cy - 0.8 * s], color=color, lw=1.8,
            solid_capstyle="round", zorder=6)


def icon_check(ax, cx, cy, r=0.075, color=GREEN):
    ax.add_patch(Circle((cx, cy), r, fc=color, ec="none", zorder=6))
    ax.plot([cx - 0.5 * r, cx - 0.12 * r, cx + 0.55 * r], [cy, cy - 0.4 * r, cy + 0.4 * r], color="white", lw=1.2,
            solid_capstyle="round", solid_joinstyle="round", zorder=7)


def icon_hospital(ax, cx, cy, s=0.2, color=BLUE):
    w, h = s, s * 0.8
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="white", ec=color, lw=1.0, zorder=6))
    ax.add_patch(Polygon([[cx - w / 2 - 0.02, cy + h / 2], [cx, cy + h / 2 + 0.06], [cx + w / 2 + 0.02, cy + h / 2]],
                         closed=True, fc=color, ec="none", zorder=6))
    c, t = 0.06, 0.02
    ax.add_patch(Rectangle((cx - t / 2, cy + 0.015 - c / 2), t, c, fc=RED, ec="none", zorder=7))
    ax.add_patch(Rectangle((cx - c / 2, cy + 0.015 - t / 2), c, t, fc=RED, ec="none", zorder=7))


def icon_shield(ax, cx, cy, s=0.2, color=BLUE):
    pts = np.array([[0, 0.5], [0.42, 0.36], [0.38, -0.08], [0, -0.5], [-0.38, -0.08], [-0.42, 0.36]]) * s
    ax.add_patch(Polygon(pts + [cx, cy], closed=True, fc=color, ec="none", zorder=6))
    ax.plot([cx - 0.16 * s, cx - 0.03 * s, cx + 0.2 * s], [cy + 0.0, cy - 0.15 * s, cy + 0.16 * s], color="white",
            lw=1.2, solid_capstyle="round", solid_joinstyle="round", zorder=7)


# ---------------------------------------------------------------------------------------------------- figure
def main() -> None:
    lanes = [lane_info(ds, i) for ds, i in LANES]
    W, H = 7.16, 2.75
    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fs = 7.0

    ys = [2.12, 1.69, 1.26, 0.83]
    s = 0.36
    x_img = 0.6
    s1_c, s1_w = 1.74, 1.24          # small-model box
    g_c, g_w = 2.97, 0.6             # risk bar
    d_c, d_w = 3.92, 0.76            # decision pill
    l_c, l_w = 5.05, 0.98            # large-model box
    r_x = 5.86                       # result icon
    top, bot = ys[0] + 0.24, ys[-1] - 0.24
    x_tau = g_c - g_w / 2 + TAU_FRAC * g_w

    # headers
    hy = top + 0.08
    for x, t in [(x_img, "Input image"), (s1_c, "Small model\n(on device)"), (g_c, "Risk of a\nmissed cancer"),
                 (d_c, "Decision"), (l_c, "Large model\n(on device)"), (r_x + 0.45, "Result")]:
        ax.text(x, hy, t, ha="center", va="bottom", fontsize=fs + 0.2, color=INK, weight="bold", linespacing=1.1)
    icon_bolt(ax, s1_c - 0.52, hy + 0.13)
    icon_magnifier(ax, l_c - 0.5, hy + 0.13)

    # dataset brackets
    groups: list[list[int]] = []
    for k, L in enumerate(lanes):
        if groups and lanes[groups[-1][0]]["ds"] == L["ds"]:
            groups[-1].append(k)
        else:
            groups.append([k])
    xb = x_img - s / 2 - 0.06
    for g in groups:
        y0, y1 = ys[g[-1]] - s / 2, ys[g[0]] + s / 2
        ax.plot([xb + 0.03, xb, xb, xb + 0.03], [y0, y0, y1, y1], color=LINE, lw=0.8)
        ax.text(xb - 0.03, (y0 + y1) / 2, lanes[g[0]]["ds"], rotation=90, ha="right", va="center",
                fontsize=fs - 1.5, color=MUTED)

    # model boxes: small model spans all lanes, large model only the escalated ones
    rbox(ax, s1_c - s1_w / 2, bot, s1_w, top - bot, NEUTRAL_FILL)
    esc_ys = [y for y, L in zip(ys, lanes) if L["esc"]]
    rbox(ax, l_c - l_w / 2, min(esc_ys) - 0.22, l_w, max(esc_ys) - min(esc_ys) + 0.44, ORANGE_FILL, ec=ORANGE)

    for y, L in zip(ys, lanes):
        col = ORANGE if L["esc"] else GREEN
        a = fig.add_axes(((x_img - s / 2) / W, (y - s / 2) / H, s / W, s / H))
        a.imshow(L["img"])
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(LINE)
        arrow(ax, x_img + s / 2 + 0.04, y, s1_c - s1_w / 2, y)

        # small model's answer, in words
        name = L["names"][L["yh"]]
        unsure = L["esc"] and not L["pred_pos"]
        ax.text(s1_c, y + 0.065, f"“{name}?”" if unsure else name, ha="center", va="center", fontsize=fs,
                color=INK)
        ax.text(s1_c, y - 0.095, f"unsure: could be {L['cancer']}" if unsure else "confident", ha="center",
                va="center", fontsize=fs - 1.4, color=ORANGE if unsure else MUTED)
        arrow(ax, s1_c + s1_w / 2, y, g_c - g_w / 2 - 0.06, y)

        # risk bar (length relative to tau; the shared tau line is drawn after the loop)
        frac = min(TAU_FRAC * L["u"] / L["tau"], 1.0)
        ax.add_patch(Rectangle((g_c - g_w / 2, y - 0.04), g_w, 0.08, fc=TRACK, ec="none", zorder=2))
        ax.add_patch(Rectangle((g_c - g_w / 2, y - 0.04), max(frac * g_w, 0.03), 0.08, fc=col, ec="none",
                               zorder=2.5))
        # tau markers just outside the bar, pointing at the threshold (the tau line itself runs behind the bars)
        m = 0.035
        ax.add_patch(Polygon([[x_tau - m, y + 0.1], [x_tau + m, y + 0.1], [x_tau, y + 0.05]], closed=True, fc=BLUE,
                             ec="none", zorder=3))
        ax.add_patch(Polygon([[x_tau - m, y - 0.1], [x_tau + m, y - 0.1], [x_tau, y - 0.05]], closed=True, fc=BLUE,
                             ec="none", zorder=3))
        arrow(ax, g_c + g_w / 2 + 0.06, y, d_c - d_w / 2, y)

        # decision pill
        rbox(ax, d_c - d_w / 2, y - 0.12, d_w, 0.24, GREEN_FILL if not L["esc"] else ORANGE_FILL, ec=col, lw=1.1,
             r=0.11, z=2)
        ax.text(d_c, y, "Escalate" if L["esc"] else "Accept", ha="center", va="center", fontsize=fs, color=col,
                weight="bold", zorder=3)

        # route to the result
        if L["esc"]:
            arrow(ax, d_c + d_w / 2, y, l_c - l_w / 2, y, color=ORANGE)
            ax.text(l_c, y, L["names"][L["yl"]], ha="center", va="center", fontsize=fs, color=INK)
            arrow(ax, l_c + l_w / 2, y, r_x - 0.12, y, color=ORANGE)
        else:
            arrow(ax, d_c + d_w / 2, y, r_x - 0.12, y, color=GREEN)
        icon_check(ax, r_x, y, color=col)
        ax.text(r_x + 0.13, y + 0.06, L["names"][L["final"]], ha="left", va="center", fontsize=fs, color=INK,
                weight="bold")
        caught = L["esc"] and L["true_pos"] and not L["pred_pos"]
        ax.text(r_x + 0.13, y - 0.09, "cancer caught" if caught else "fast path", ha="left", va="center",
                fontsize=fs - 1.4, color=col)

    ax.text((d_c + d_w / 2 + r_x - 0.12) / 2, ys[0] + 0.045, "most images stop here", ha="center", va="bottom",
            fontsize=fs - 1.5, color=GREEN, style="italic")

    # calibration strip
    yc, hc = 0.05, 0.32
    steps = [(0.05, 1.95, "Labelled images from the\nhospital, collected once"),
             (2.25, 1.75, "Paired Learn-then-Test\non both models"),
             (4.25, 2.86, "Threshold $\\tau$ keeps ≥ 97% of ResNet-18's\nsensitivity, with 95% confidence")]
    for i, (x0, w, t) in enumerate(steps):
        rbox(ax, x0, yc, w, hc, BLUE_FILL, ec=BLUE, ls="--", lw=0.9)
        tx = x0 + (0.42 if i != 1 else w / 2)
        ax.text(tx, yc + hc / 2, t, ha="left" if i != 1 else "center", va="center", fontsize=fs - 1.0, color=INK,
                linespacing=1.15)
        if i:
            px0, pw, _ = steps[i - 1]
            arrow(ax, px0 + pw, yc + hc / 2, x0, yc + hc / 2, color=BLUE)
    icon_hospital(ax, steps[0][0] + 0.21, yc + hc / 2 - 0.01)
    icon_shield(ax, steps[2][0] + 0.21, yc + hc / 2)

    # one shared tau line: from the guarantee up through every risk bar
    y_mid = (yc + hc + bot) / 2
    x_from = steps[2][0] + 0.21
    ax.plot([x_from, x_from, x_tau, x_tau], [yc + hc, y_mid, y_mid, top - 0.02], color=BLUE, ls=(0, (3, 2)),
            lw=1.0, zorder=1.5)  # below the bars (zorder 2-2.5): visible only between lanes
    ax.text(x_tau + 0.04, top - 0.04, r"$\tau$", ha="left", va="top", fontsize=fs + 0.6, color=BLUE, zorder=6)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"pipeline.{ext}", dpi=300)
    print("wrote", OUT / "pipeline.pdf")
    for L in lanes:
        print(f"{L['ds']:15s} true={L['names'][L['y']]:13s} small={L['names'][L['yh']]:13s} "
              f"p_pos={L['p_pos']:.2f} u={L['u']:.2f} tau={L['tau']:.2f} {'ESC' if L['esc'] else 'acc'} "
              f"final={L['names'][L['final']]}")


if __name__ == "__main__":
    main()
