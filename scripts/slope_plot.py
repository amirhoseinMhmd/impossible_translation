"""
Slope chart (before/after) comparing input vs prediction perplexity.
Left axis = Input (perturbed), right axis = Prediction (model output).
One line per perturbation type.

Loads from perplexity_results/*.json (produced by compute_perplexity_csv.py).

Usage:
    python3 scripts/slope_plot.py
    python3 scripts/slope_plot.py --corpus gutenberg --output paper_plots/perplexity_slope
"""

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# ACL style
# ---------------------------------------------------------------------------
COL_W = 3.2
COL_H = 3.4

def setup_acl_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.7,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.bottom": False,
        "axes.spines.left": True,
        "axes.spines.right": True,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.8",
        "legend.fancybox": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.35,
    })


# ---------------------------------------------------------------------------
# Colors — one per perturbation
# ---------------------------------------------------------------------------
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
BASELINE_COLOR = "#444444"

PERTURBATION_LABELS = {
    "fullShuffle":    "Full Shuffle",
    "localShuffle3":  "Local Shuffle (w=3)",
    "partialReverse": "Partial Reverse",
    "phraseShuffle":  "Phrase Shuffle",
}


# ---------------------------------------------------------------------------
# Data loading (same as dumbbell_plot.py)
# ---------------------------------------------------------------------------
def load_data(results_dir="perplexity_results", corpus_filter="gutenberg"):
    pattern = f"{results_dir}/{corpus_filter}_*_perplexity.json"
    files = sorted(glob.glob(pattern))
    if not files:
        files = sorted(glob.glob(f"{results_dir}/*_perplexity.json"))

    rows = []
    baseline = None

    for fpath in files:
        fname = Path(fpath).stem
        m = re.search(r"_([^_]+)_perplexity$", fname)
        if not m:
            continue
        perturbation = m.group(1)
        label = PERTURBATION_LABELS.get(perturbation, perturbation)

        with open(fpath) as f:
            raw = json.load(f)

        all_input, all_pred, all_actual = [], [], []
        for ckpt_data in raw.values():
            all_input.extend(ckpt_data.get("input", []))
            all_pred.extend(ckpt_data.get("prediction", []))
            all_actual.extend(ckpt_data.get("actual", []))

        if not all_pred:
            continue

        rows.append({
            "label":     label,
            "input":     float(np.mean(all_input)),
            "pred":      float(np.mean(all_pred)),
            "pred_std":  float(np.std(all_pred, ddof=1)) if len(all_pred) > 1 else 0.0,
        })

        if baseline is None and all_actual:
            baseline = float(np.mean(all_actual))

    rows.sort(key=lambda r: r["input"], reverse=True)
    return rows, baseline


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_slope(rows, baseline, output="paper_plots/perplexity_slope"):
    setup_acl_style()

    fig, ax = plt.subplots(figsize=(COL_W, COL_H))

    x_input = 0.0
    x_pred  = 1.0

    # Draw baseline as horizontal reference on both sides
    if baseline is not None:
        for x in [x_input, x_pred]:
            ax.scatter(x, baseline, color=BASELINE_COLOR, s=30,
                       zorder=5, marker="D")
        ax.plot([x_input, x_pred], [baseline, baseline],
                color=BASELINE_COLOR, linewidth=1.0, linestyle="--",
                alpha=0.6, zorder=2)
        # Label on right
        ax.text(x_pred + 0.04, baseline, f"Original\n({baseline:.0f})",
                va="center", ha="left", fontsize=7.5,
                color=BASELINE_COLOR)

    # Draw one slope line per perturbation
    for i, r in enumerate(rows):
        color = COLORS[i % len(COLORS)]

        # Connecting line
        ax.plot([x_input, x_pred], [r["input"], r["pred"]],
                color=color, linewidth=1.6, alpha=0.85, zorder=3,
                solid_capstyle="round")

        # Dots
        ax.scatter(x_input, r["input"], color=color, s=50, zorder=5)
        ax.scatter(x_pred,  r["pred"],  color=color, s=50, zorder=5)

        # Std error bar on prediction dot
        if r["pred_std"] > 0:
            ax.errorbar(x_pred, r["pred"], yerr=r["pred_std"],
                        fmt="none", ecolor=color, elinewidth=1.0,
                        capsize=3, capthick=0.8, alpha=0.7, zorder=4)

        # Labels — input on left, prediction on right
        ax.text(x_input - 0.04, r["input"], f"{r['input']:.0f}",
                va="center", ha="right", fontsize=7.5, color=color)
        ax.text(x_pred + 0.04, r["pred"], f"{r['pred']:.0f}",
                va="center", ha="left", fontsize=7.5, color=color)

    # Axis formatting
    ax.set_yscale("log")
    ax.set_xlim(-0.35, 1.55)
    ax.set_xticks([x_input, x_pred])
    ax.set_xticklabels(["Input\n(perturbed)", "Prediction"], fontsize=9)
    ax.set_ylabel("Average Perplexity (log scale)")
    ax.tick_params(bottom=False, left=True)

    # Spine — keep only left and right
    ax.spines["left"].set_bounds(
        min(r["pred"] for r in rows) * 0.85,
        max(r["input"] for r in rows) * 1.1,
    )
    ax.spines["right"].set_position(("data", x_pred))
    ax.spines["right"].set_bounds(
        min(r["pred"] for r in rows) * 0.85,
        max(r["input"] for r in rows) * 1.1,
    )

    # Legend (perturbation colors)
    handles = [
        mpatches.Patch(color=COLORS[i % len(COLORS)], label=r["label"])
        for i, r in enumerate(rows)
    ]
    if baseline is not None:
        handles.append(mpatches.Patch(color=BASELINE_COLOR,
                                      label=f"Original ({baseline:.0f})"))
    ax.legend(handles=handles, loc="upper right",
              fontsize=7.5, ncol=1, bbox_to_anchor=(1.0, 1.0))

    fig.tight_layout()

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}.png / .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="perplexity_results")
    parser.add_argument("--corpus", default="gutenberg")
    parser.add_argument("--output", default="paper_plots/perplexity_slope")
    args = parser.parse_args()

    rows, baseline = load_data(args.results_dir, args.corpus)
    if not rows:
        raise FileNotFoundError(
            f"No perplexity JSON files found in '{args.results_dir}/'. "
            "Run compute_perplexity_csv.py first."
        )

    print(f"Loaded {len(rows)} perturbation(s):")
    for r in rows:
        print(f"  {r['label']}: input={r['input']:.1f}  "
              f"pred={r['pred']:.1f} (±{r['pred_std']:.1f})")
    if baseline:
        print(f"  Baseline (original): {baseline:.2f}")

    plot_slope(rows, baseline, output=args.output)


if __name__ == "__main__":
    main()
