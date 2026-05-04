"""
EM/BLEU Plot — ACL/EMNLP Academic Style
Matches the dependency analysis plot style exactly.

Usage:
    python plot_em_bleu.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pathlib import Path


# ══════════════════════════════════════════════════
# ACL STYLE — identical to dependency plots
# ══════════════════════════════════════════════════
COL_W = 4  # inches
COL_H = 4   # inches
# DOUBLE_W = 6.75  # full page width

def setup_acl_style():
    plt.rcParams.update({
        # LaTeX-matching serif fonts
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",

        # Font sizes — ACL 2-column
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        # Figure
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 5,

        # Axes
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.linewidth": 0.3,
        "grid.alpha": 0.35,
        "grid.linestyle": "-",

        # Ticks
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.pad": 3,
        "ytick.major.pad": 3,

        # Spines
        "axes.spines.top": True,
        "axes.spines.right": True,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.7",
        "legend.fancybox": False,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.4,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.3,
    })


# ══════════════════════════════════════════════════
# COLORS & MARKERS — high contrast, colorblind-safe
# ══════════════════════════════════════════════════
COLORS = [
    "#FF8B05",   # orange
    "#2EB3FF",   # sky blue
    "#98DE2C",   # lime green
    "#FF5757",   # red
    "#8E7DFF",   # purple
    "#008F27",   # dark green
    "#FFBE4F",   # gold
    "#DBDB00",   # yellow
    "#FF6E74",   # salmon
]

MARKERS = ["o", "s", "^", "v", "D", "p", "*", "X", "P", "h"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]


# ══════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════
def extract_checkpoint_number(checkpoint_name):
    if checkpoint_name == "final":
        return float('inf')
    try:
        return int(checkpoint_name.split('-')[1])
    except (IndexError, ValueError):
        return 0


def load_json_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    items = [(k, v) for k, v in data.items() if k != "final"]
    items.sort(key=lambda x: extract_checkpoint_number(x[0]))

    checkpoint_nums = [extract_checkpoint_number(k) for k, v in items]
    values = [v for k, v in items]
    final_value = data.get("final")

    return checkpoint_nums, values, final_value


# ══════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════
def plot_em_bleu(json_files, labels=None, title="",
                 xlabel="Checkpoint", ylabel="Average Score",
                 figsize=None, save_name="em_bleu_plot.png",
                 use_cbrt_scale=True):
    """
    Plot EM/BLEU in ACL style matching dependency analysis plots.
    """

    if figsize is None:
        figsize = (COL_W, COL_H)  # square single-column

    if labels is None:
        labels = [Path(f).stem.split('_')[4].title() for f in json_files]

    fig, ax = plt.subplots(figsize=figsize)

    for idx, (json_file, label) in enumerate(zip(json_files, labels)):
        checkpoints, values, final_value = load_json_data(json_file)

        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        linestyle = LINESTYLES[idx % len(LINESTYLES)]

        ax.plot(checkpoints, values,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=5,
                markeredgewidth=0.5,
                markeredgecolor='white',
                label=label,
                alpha=0.9)

    if use_cbrt_scale:
        ax.set_yscale("function", functions=(np.cbrt, lambda x: x ** 3))
        ax.set_yticks(np.arange(0, 1.0, 0.1))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    else:
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

    ax.set_ylim(0, 1)
    # ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.legend(loc='best')
    ax.margins(x=0.02, y=0.05)

    fig.tight_layout()
    fig.savefig(save_name, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_name}")

    # Also save PDF if png requested
    if save_name.endswith('.png'):
        pdf_name = save_name.replace('.png', '.pdf')
        fig.savefig(pdf_name, dpi=400, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {pdf_name}")

    plt.close(fig)


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    setup_acl_style()

    json_files = [
        'results/results_gutenberg-100k-localShuffle-exact_match.json',
        'results/results_gutenberg-100k-partialReverse-exact_match.json',
        'results/results_gutenberg-100k-wordHop-exact_match.json',
        'results/results_gutenberg-100k-localShuffle-blue.json',
        'results/results_gutenberg-100k-partialReverse-blue.json',
        'results/results_gutenberg-100k-wordHop-blue.json',
    ]

    labels = [
        'local shuffle EM', 'partial reverse EM', 'word hop EM',
        'local shuffle BLEU', 'partial reverse BLEU', 'word hop BLEU',
    ]

    plot_em_bleu(
        json_files=json_files,
        labels=labels,
        title='',
        xlabel='Checkpoint',
        ylabel='Average Score',
        figsize=(COL_W, COL_H),
        save_name='gutenberg_100k_plot.png',
        use_cbrt_scale=True,
    )