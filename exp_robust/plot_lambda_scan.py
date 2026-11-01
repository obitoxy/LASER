"""
plot_lambda_scan.py — λ-ACC-V curve: VGG-16 (this scan) vs ResNeXt-50
(paper Appendix B), VGG-16/SB=1/CIFAR-10, victim truck→car.

Reads exp_robust/outputs/results_lambda_scan.json for the VGG-16 curve and
hardcodes the ResNeXt-50 values from the paper's Appendix B (Tab. lambda_impact).
Writes lambda_scan_accv.{pdf,png} into exp_robust/outputs/figs/.
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# --- palette (validated, dataviz skill: categorical slots 1/2, light mode) ---
BLUE = "#2a78d6"   # VGG-16
GREEN = "#008300"  # ResNeXt-50
INK = "#0b0b0b"
SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# ResNeXt-50 ACC-V from paper Appendix B (Tab. lambda_impact)
RESNEXT = {0.50: 75.90, 0.55: 56.70, 0.57: 33.10,
           0.59: 19.40, 0.80: 0.50, 1.00: 0.00}


def main():
    with open(os.path.join(OUT, "results_lambda_scan.json")) as f:
        d = json.load(f)
    vgg = {r["lambda"]: r["acc_v"] for r in d["rows"]}
    lam = sorted(vgg.keys())

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "xtick.color": SEC,
        "ytick.color": SEC,
        "text.color": INK,
        "axes.linewidth": 1.0,
    })

    fig, ax = plt.subplots(figsize=(4.8, 3.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xs = lam
    ys_vgg = [vgg[x] for x in xs]
    ys_rx = [RESNEXT[x] for x in xs]

    # VGG-16: blue, circle markers
    ax.plot(xs, ys_vgg, color=BLUE, lw=2, marker="o", ms=6.5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
            label="VGG-16 (SB=1, this work)")
    # ResNeXt-50: green, square markers (shape = secondary encoding for print/CVD)
    ax.plot(xs, ys_rx, color=GREEN, lw=2, marker="s", ms=6.0,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
            label="ResNeXt-50 (paper App. B)")

    # recessive y gridlines only
    ax.yaxis.grid(True, color=GRID, lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.2f}" for x in xs])
    ax.set_xlim(0.485, 1.015)
    ax.set_ylim(-2, 82)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlabel("Steering coefficient $\\lambda$", labelpad=4)
    ax.set_ylabel("Victim Accuracy  ACC-V (%)", labelpad=4)

    leg = ax.legend(loc="upper right", frameon=True, framealpha=0.95,
                    edgecolor=GRID, fontsize=8.5, handlelength=2.2,
                    borderpad=0.5)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    figs = os.path.join(OUT, "figs")
    os.makedirs(figs, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(figs, f"lambda_scan_accv.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor=SURFACE)
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
