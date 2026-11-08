"""
Dumbbell chart comparing input vs prediction perplexity per perturbation.
Loads data from perplexity_results/*.json (produced by compute_perplexity_csv.py).

Usage:
    python3 scripts/dumbbell_plot.py
    python3 scripts/dumbbell_plot.py --corpus gutenberg --output paper_plots/perplexity_dumbbell
"""

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# ACL style
# ---------------------------------------------------------------------------
COL_W = 5.0
COL_H = 4.0

def setup_acl_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.8,
        "lines.markersize": 7,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "grid.linestyle": "-",
        "grid.color": "#cccccc",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.pad": 4,
        "ytick.major.pad": 4,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.75",
        "legend.fancybox": False,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.4,
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.3,
    })


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
INPUT_COLOR    = "#c71022"   # blue  — perturbed input
PRED_COLOR     = "#008c1c"   # vermillion — model prediction
BASELINE_COLOR = "#444444"   # dark gray — original text baseline
CONNECTOR_COLOR = "#aaaaaa"  # gray connector line


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
PERTURBATION_LABELS = {
    "fullShuffle":    "Full Shuffle",
    "localShuffle3":  "Local Shuffle (w=3)",
    "partialReverse": "Partial Reverse",
    "phraseShuffle":  "Phrase Shuffle",
}


# ---------------------------------------------------------------------------
# Data loading
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
            "input_std": float(np.std(all_input, ddof=1)) if len(all_input) > 1 else 0.0,
        })

        if baseline is None and all_actual:
            baseline = float(np.mean(all_actual))

    # Sort by input perplexity descending so hardest perturbation is on top
    rows.sort(key=lambda r: r["input"], reverse=True)
    return rows, baseline


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_dumbbell(rows, baseline, output="perplexity_dumbbell"):
    setup_acl_style()

    n = len(rows)
    x = np.arange(n)
    labels = [r["label"] for r in rows]

    fig, ax = plt.subplots(figsize=(max(COL_W, n * 1.1 + 0.8), COL_H))

    # Baseline horizontal line
    if baseline is not None:
        ax.axhline(baseline, color=BASELINE_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.7, zorder=1,
                   label=f"Original ({baseline:.0f})")

    # Connector lines (drawn first so dots sit on top)
    for i, r in enumerate(rows):
        ax.plot([x[i], x[i]], [r["input"], r["pred"]],
                color=CONNECTOR_COLOR, linewidth=2.5, zorder=2,
                solid_capstyle="round")

    # Input dots
    input_vals = [r["input"] for r in rows]
    input_stds = [r["input_std"] for r in rows]
    ax.scatter(x, input_vals, color=INPUT_COLOR, s=65, zorder=4,
               label="Input (perturbed)")
    ax.errorbar(x, input_vals, yerr=input_stds,
                fmt="none", ecolor=INPUT_COLOR, elinewidth=1.0,
                capsize=3, capthick=0.8, alpha=0.7, zorder=3)

    # Prediction dots
    pred_vals = [r["pred"] for r in rows]
    pred_stds = [r["pred_std"] for r in rows]
    ax.scatter(x, pred_vals, color=PRED_COLOR, s=65, zorder=4,
               label="Prediction")
    ax.errorbar(x, pred_vals, yerr=pred_stds,
                fmt="none", ecolor=PRED_COLOR, elinewidth=1.0,
                capsize=3, capthick=0.8, alpha=0.7, zorder=3)

    # Value annotations above/below each dot
    for i, r in enumerate(rows):
        ax.text(x[i] + 0.07, r["input"], f"{r['input']:.0f}",
                color=INPUT_COLOR, fontsize=7.5, va="center", ha="left")
        ax.text(x[i] + 0.07, r["pred"], f"{r['pred']:.0f}",
                color=PRED_COLOR, fontsize=7.5, va="center", ha="left")

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel("Average Perplexity (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha="center")
    ax.set_xlim(-0.5, n - 0.5)
    ax.tick_params(bottom=False)
    ax.grid(axis="y", alpha=0.35, linewidth=0.4)
    ax.grid(axis="x", visible=False)

    ax.legend(loc="upper right", fontsize=8, ncol=1)

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
    parser.add_argument("--output", default="paper_plots/perplexity_dumbbell")
    args = parser.parse_args()

    rows, baseline = load_data(args.results_dir, args.corpus)
    if not rows:
        raise FileNotFoundError(
            f"No perplexity JSON files found in '{args.results_dir}/'. "
            "Run compute_perplexity_csv.py first."
        )

    print(f"Loaded {len(rows)} perturbation(s):")
    for r in rows:
        print(f"  {r['label']}: input={r['input']:.1f} (±{r['input_std']:.1f})  "
              f"pred={r['pred']:.1f} (±{r['pred_std']:.1f})")
    if baseline:
        print(f"  Baseline (original): {baseline:.2f}")

    plot_dumbbell(rows, baseline, output=args.output)


if __name__ == "__main__":
    main()
