"""
Generate all paper plots from evaluation_results/.

Reads evaluation_results/{corpus}/{perturbation}/{seed}/ directories,
averages across seeds, and produces ACL-style plots with error bars (min/max shading).

For phraseShuffle (single seed), no error bars are shown.

Usage:
    python scripts/generate_paper_plots.py
    python scripts/generate_paper_plots.py --output-dir paper_plots
    python scripts/generate_paper_plots.py --error-bar std   # std | sem | minmax (default)
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

# ---------------------------------------------------------------------------
# ACL style
# ---------------------------------------------------------------------------
COL_W = 8.0
COL_H = 6.0
DOUBLE_W = 10.0

def setup_acl_style():
    plt.rcParams.update({
        # Sans-serif font (matches reference image)
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",

        # Font sizes
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 7,

        # Axes
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "grid.linestyle": "-",
        "grid.color": "#cccccc",

        # Ticks
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.pad": 4,
        "ytick.major.pad": 4,

        # All 4 spines visible (matches reference)
        "axes.spines.top": True,
        "axes.spines.right": True,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.75",
        "legend.fancybox": False,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.4,
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.3,
    })

LINE_WIDTH = 1.8   # global line thickness — change this to adjust all plots

# Colorblind-friendly, high-contrast palette matching reference image
COLORS = [
    "#555555",  # dark gray  (NoShuffle / baseline style)
    "#d62728",  # red
    "#ff7f0e",  # orange/gold
    "#bcbd22",  # olive/yellow-green
    "#2ca02c",  # green
    "#17becf",  # cyan/teal
    "#1f77b4",  # blue
    "#9467bd",  # purple
    "#e377c2",  # pink
    "#8c564b",  # brown
]

# Distinct marker per line — matches reference (circles, squares, triangles, etc.)
MARKERS = ["o", "s", "^", "v", "D", "p", "X", "*", "P", "h"]

# Mix of solid and dashed linestyles — matches reference
LINESTYLES = ["-", "-", "-", "--", "-.", "-", "--", "-.", ":", "-"]

PERTURBATION_LABELS = {
    "partialReverse": "Partial Reverse",
    "localShuffle3": "Local Shuffle (w=3)",
    "localShuffle5": "Local Shuffle (w=5)",
    "localShuffle": "Local Shuffle",
    "fullShuffle": "Full Shuffle",
    "wordHop": "Word Hop",
    "phraseShuffle": "Phrase Shuffle",
}

CORPUS_LABELS = {
    "gutenberg": "Gutenberg",
    "bnc_spoken": "BNC Spoken",
}

METRIC_FILES = {
    "BLEU": "BLEU",
    "exact_match": "exact_match",
    "dep_f1": "dep_f1",
    "avg_dependency_length": "avg_dependency_length",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def checkpoint_sort_key(name):
    if name == "final":
        return float("inf")
    try:
        return int(name.split("-")[1])
    except (IndexError, ValueError):
        return 0


def discover_experiments(results_dir):
    """
    Returns list of dicts:
      {corpus, perturbation, seed, path}
    """
    results_dir = Path(results_dir)
    experiments = []

    # Standard layout: {corpus}/{perturbation}/{seed}/
    for corpus_dir in sorted(results_dir.iterdir()):
        if not corpus_dir.is_dir():
            continue
        corpus = corpus_dir.name

        # Check if this is the phraseShuffle special case (no perturbation subdir)
        seed_dirs = [d for d in corpus_dir.iterdir() if d.is_dir()]
        has_metric_json = any(
            f.name.endswith("_BLEU.json") or f.name.endswith("_exact_match.json")
            for d in seed_dirs
            for f in d.iterdir()
            if f.is_file()
        ) if seed_dirs else False

        if has_metric_json and not any(
            (d / next(d.iterdir()).name if any(d.iterdir()) else d).is_dir()
            for d in seed_dirs
            if any(d.iterdir())
        ):
            # Flat layout: {perturbation}/{seed}/ (e.g. phraseShuffle/42/)
            perturbation = corpus  # dir name is the perturbation
            for seed_dir in seed_dirs:
                # Infer corpus from filenames, e.g.
                # results_1k_gutenberg_phraseShuffle_BLEU.json -> gutenberg
                inferred_corpus = perturbation  # fallback
                bleu_files = list(seed_dir.glob("results_*_BLEU.json"))
                if bleu_files:
                    fname = bleu_files[0].stem  # results_1k_gutenberg_phraseShuffle_BLEU
                    body = fname.replace("results_", "").replace(f"_{perturbation}_BLEU", "")
                    # body = "1k_gutenberg" -> extract last part as corpus
                    parts = body.split("_")
                    if len(parts) >= 2:
                        inferred_corpus = parts[-1]  # "gutenberg"
                    else:
                        inferred_corpus = body
                experiments.append({
                    "corpus": inferred_corpus,
                    "perturbation": perturbation,
                    "seed": seed_dir.name,
                    "path": seed_dir,
                })
        else:
            # Standard layout
            for pert_dir in sorted(corpus_dir.iterdir()):
                if not pert_dir.is_dir():
                    continue
                perturbation = pert_dir.name
                for seed_dir in sorted(pert_dir.iterdir()):
                    if not seed_dir.is_dir():
                        continue
                    experiments.append({
                        "corpus": corpus,
                        "perturbation": perturbation,
                        "seed": seed_dir.name,
                        "path": seed_dir,
                    })

    return experiments


def load_metric_json(seed_path, metric_suffix):
    """Load a results_*_{metric_suffix}.json file from seed_path."""
    matches = list(Path(seed_path).glob(f"results_*_{metric_suffix}.json"))
    if not matches:
        return None
    with open(matches[0], "r") as f:
        return json.load(f)


def load_summary_json(seed_path):
    """Load the summary JSON from seed_path."""
    matches = list(Path(seed_path).glob("results_*_summary.json"))
    if not matches:
        return None
    with open(matches[0], "r") as f:
        return json.load(f)


def aggregate_metric_across_seeds(experiments, metric_suffix):
    """
    Groups experiments by (corpus, perturbation), loads metric JSON for each seed,
    and returns:
        {(corpus, perturbation): {checkpoint_num: [values_across_seeds]}}
    """
    grouped = defaultdict(lambda: defaultdict(list))

    for exp in experiments:
        key = (exp["corpus"], exp["perturbation"])
        data = load_metric_json(exp["path"], metric_suffix)
        if data is None:
            continue
        for ckpt_name, value in data.items():
            if ckpt_name == "final":
                continue
            ckpt_num = checkpoint_sort_key(ckpt_name)
            grouped[key][ckpt_num].append(float(value))

    return grouped


def aggregate_summary_across_seeds(experiments):
    """
    Groups by (corpus, perturbation), loads summary JSON for each seed,
    returns: {(corpus, perturbation): {checkpoint_num: {metric: [values]}}}
    """
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for exp in experiments:
        key = (exp["corpus"], exp["perturbation"])
        data = load_summary_json(exp["path"])
        if data is None:
            continue
        for ckpt_name, ckpt_data in data.items():
            if ckpt_name == "final":
                continue
            ckpt_num = checkpoint_sort_key(ckpt_name)
            if isinstance(ckpt_data, dict):
                for metric, value in ckpt_data.items():
                    if isinstance(value, (int, float)):
                        grouped[key][ckpt_num][metric].append(float(value))

    return grouped


def aggregate_csv_across_seeds(experiments):
    """
    Groups by (corpus, perturbation), loads per-checkpoint CSVs for each seed.
    Returns: {(corpus, perturbation): {checkpoint_num: {col: [seed_means]}}}
    """
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for exp in experiments:
        key = (exp["corpus"], exp["perturbation"])
        csv_files = sorted(Path(exp["path"]).glob("full_samples_*_checkpoint-*.csv"))
        for csv_file in csv_files:
            import re
            m = re.search(r"checkpoint-(\d+)", csv_file.name)
            if not m:
                continue
            ckpt_num = int(m.group(1))
            df = pd.read_csv(csv_file)
            for col in ["adl_input", "adl_prediction", "adl_actual", "dep_f1",
                         "dep_precision", "dep_recall", "baseline_f1"]:
                if col in df.columns:
                    valid = df[col].dropna()
                    if not valid.empty:
                        grouped[key][ckpt_num][col].append(float(valid.mean()))

    # The per-checkpoint adl_input/baseline_f1 columns came from a word mapping that
    # misaligns subword-perturbed inputs; use the permutation-based values instead.
    override = Path(__file__).resolve().parents[1] / "evaluation_results" / "rq_v2" / "input_baselines.csv"
    if override.exists():
        ob = pd.read_csv(override).set_index(["dataset", "perturbation"])
        for key, ckpts in grouped.items():
            if key in ob.index:
                for cols in ckpts.values():
                    cols["adl_input"] = [float(ob.loc[key, "adl_input"])]
                    cols["baseline_f1"] = [float(ob.loc[key, "baseline_f1"])]

    return grouped


def compute_stats(values):
    """Returns mean, std, sem, min, max for a list of values."""
    arr = np.array(values)
    mean = float(np.mean(arr))
    if len(arr) > 1:
        std = float(np.std(arr, ddof=1))
        sem = std / np.sqrt(len(arr))
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
    else:
        std = sem = 0.0
        vmin = vmax = mean
    return mean, std, sem, vmin, vmax


def get_error(mean, std, sem, vmin, vmax, error_bar):
    if error_bar == "std":
        return mean - std, mean + std
    elif error_bar == "sem":
        return mean - sem, mean + sem
    elif error_bar == "minmax":
        return vmin, vmax
    return mean, mean


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    if path.suffix == ".png":
        fig.savefig(path.with_suffix(".pdf"), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_series(ax, checkpoints, means, errs, color, marker, linestyle,
                 label, has_multi_seeds, elinewidth=1.5):
    """Plot a single line with vertical error bars (reference style)."""
    means = np.array(means, dtype=float)
    errs = np.array(errs, dtype=float)

    ax.plot(checkpoints, means,
            color=color, marker=marker, linestyle=linestyle,
            linewidth=LINE_WIDTH, markersize=7,
            markeredgewidth=0.8, markeredgecolor=color,
            markerfacecolor=color,
            label=label, zorder=3)

    if has_multi_seeds and np.any(errs > 0):
        ax.errorbar(checkpoints, means, yerr=errs,
                     fmt="none", ecolor=color, elinewidth=elinewidth,
                     capsize=4, capthick=1.0, alpha=0.85, zorder=2)


def _compute_series(ckpt_data, error_bar):
    """Compute checkpoints, means, error values from {ckpt: [values]}."""
    checkpoints = sorted(ckpt_data.keys())
    means, errs = [], []
    has_multi = False
    for ckpt in checkpoints:
        vals = ckpt_data[ckpt]
        mean, std, sem, vmin, vmax = compute_stats(vals)
        means.append(mean)
        if len(vals) > 1:
            has_multi = True
            if error_bar == "std":
                errs.append(std)
            elif error_bar == "sem":
                errs.append(sem)
            else:  # minmax
                errs.append(max(mean - vmin, vmax - mean))
        else:
            errs.append(0.0)
    return checkpoints, means, errs, has_multi


# ---------------------------------------------------------------------------
# Plot functions (reference style: vertical error bars, linear scale)
# ---------------------------------------------------------------------------
def plot_checkpoint_curves(grouped_data, metric_label, output_path,
                           error_bar="std", ylim=(0, 1),
                           use_cbrt_scale=False, corpus_title="",
                           show_legend=False):
    """
    Plot checkpoint curves for all perturbations within a corpus.
    grouped_data: {perturbation: {ckpt: [values]}}
    """
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))

    for idx, (perturbation, ckpt_data) in enumerate(sorted(grouped_data.items())):
        checkpoints, means, errs, has_multi = _compute_series(ckpt_data, error_bar)
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        linestyle = LINESTYLES[idx % len(LINESTYLES)]
        label = PERTURBATION_LABELS.get(perturbation, perturbation)
        _plot_series(ax, checkpoints, means, errs, color, marker, linestyle,
                     label, has_multi)

    if ylim is not None:
        ax.set_ylim(*ylim)
    if show_legend:
        ax.legend(loc="best", ncol=1, frameon=True)
    ax.margins(x=0.02, y=0.05)

    fig.tight_layout()
    _save(fig, output_path)


def plot_em_bleu_combined(grouped_em, grouped_bleu, output_path,
                          error_bar="std", corpus_title="",
                          show_legend=False):
    """
    Plot EM and BLEU together for all perturbations (same axes).
    Each (perturbation, metric) gets its own color, marker, and linestyle.
    """
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))

    perturbations = sorted(set(grouped_em.keys()) | set(grouped_bleu.keys()))

    # Build series list: interleave EM/BLEU per perturbation so columns align
    series = []
    for perturbation in perturbations:
        label_base = PERTURBATION_LABELS.get(perturbation, perturbation)
        em_data = grouped_em.get(perturbation, {})
        if em_data:
            series.append((em_data, f"{label_base} EM"))
        bleu_data = grouped_bleu.get(perturbation, {})
        if bleu_data:
            series.append((bleu_data, f"{label_base} BLEU"))

    for idx, (metric_data, label) in enumerate(series):
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        linestyle = LINESTYLES[idx % len(LINESTYLES)]
        checkpoints, means, errs, has_multi = _compute_series(metric_data, error_bar)
        _plot_series(ax, checkpoints, means, errs, color, marker, linestyle,
                     label, has_multi)

    ax.set_yscale("function", functions=(np.cbrt, lambda x: x ** 3))
    ax.set_yticks(np.arange(0, 1.0, 0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.set_ylim(0, 1)
    ax.margins(x=0.02, y=0.05)

    if show_legend:
        ax.legend(loc="lower center", ncol=1, frameon=True)

    fig.tight_layout()
    _save(fig, output_path)


def plot_adl_curves(csv_grouped, output_path, error_bar="std", corpus_title=""):
    """
    Plot ADL (prediction vs input vs actual reference) per perturbation.
    csv_grouped: {perturbation: {ckpt: {col: [seed_means]}}}
    """
    for pert_idx, (perturbation, ckpt_data) in enumerate(sorted(csv_grouped.items())):
        fig, ax = plt.subplots(figsize=(COL_W, COL_H))

        checkpoints = sorted(ckpt_data.keys())
        pred_means, pred_errs = [], []
        input_refs, actual_refs = [], []
        has_multi = False

        for ckpt in checkpoints:
            cols = ckpt_data[ckpt]
            if "adl_prediction" in cols:
                mean, std, sem, vmin, vmax = compute_stats(cols["adl_prediction"])
                pred_means.append(mean)
                if len(cols["adl_prediction"]) > 1:
                    has_multi = True
                    pred_errs.append(std if error_bar == "std" else
                                     sem if error_bar == "sem" else
                                     max(mean - vmin, vmax - mean))
                else:
                    pred_errs.append(0.0)
            if "adl_input" in cols:
                input_refs.extend(cols["adl_input"])
            if "adl_actual" in cols:
                actual_refs.extend(cols["adl_actual"])

        color = COLORS[pert_idx % len(COLORS)]
        marker = MARKERS[pert_idx % len(MARKERS)]
        label = PERTURBATION_LABELS.get(perturbation, perturbation)

        if actual_refs:
            actual_mean = float(np.mean(actual_refs))
            ax.axhline(y=actual_mean, color="#222222", linestyle="--",
                        linewidth=1.2, label="Original", zorder=1)
        if input_refs:
            input_mean = float(np.mean(input_refs))
            ax.axhline(y=input_mean, color=color, linestyle=":",
                        linewidth=1.1, alpha=0.7, label="Input (perturbed)", zorder=1)

        _plot_series(ax, checkpoints, pred_means, [0.0] * len(pred_errs),
                     color, marker, "-", "Prediction", False)

        ax.legend(loc="best")
        ax.margins(x=0.02, y=0.05)

        fig.tight_layout()
        stem = Path(output_path).stem
        parent = Path(output_path).parent
        _save(fig, parent / f"{stem}_{perturbation}.png")


def plot_adl_multipanel(csv_grouped, output_path, error_bar="std", corpus_title="",
                        show_legend=False):
    """
    Single ADL plot with broken y-axis: bottom panel shows prediction curves
    (zoomed in), top panel shows the spread-out input reference lines.
    """
    perturbations = sorted(csv_grouped.keys())
    if not perturbations:
        return

    # Collect all data first to determine axis breaks
    series_data = []
    for idx, perturbation in enumerate(perturbations):
        ckpt_data = csv_grouped[perturbation]
        checkpoints = sorted(ckpt_data.keys())
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        linestyle = LINESTYLES[idx % len(LINESTYLES)]
        label = PERTURBATION_LABELS.get(perturbation, perturbation)

        pred_means, pred_errs = [], []
        input_refs, actual_refs = [], []
        has_multi = False

        for ckpt in checkpoints:
            cols = ckpt_data[ckpt]
            if "adl_prediction" in cols:
                mean, std, sem, vmin, vmax = compute_stats(cols["adl_prediction"])
                pred_means.append(mean)
                if len(cols["adl_prediction"]) > 1:
                    has_multi = True
                    pred_errs.append(std if error_bar == "std" else
                                     sem if error_bar == "sem" else
                                     max(mean - vmin, vmax - mean))
                else:
                    pred_errs.append(0.0)
            if "adl_input" in cols:
                input_refs.extend(cols["adl_input"])
            if "adl_actual" in cols:
                actual_refs.extend(cols["adl_actual"])

        input_mean = float(np.mean(input_refs)) if input_refs else None
        actual_mean = float(np.mean(actual_refs)) if actual_refs else None

        series_data.append({
            "perturbation": perturbation,
            "checkpoints": checkpoints,
            "pred_means": pred_means,
            "pred_errs": pred_errs,
            "has_multi": has_multi,
            "input_mean": input_mean,
            "actual_mean": actual_mean,
            "color": color,
            "marker": marker,
            "linestyle": linestyle,
            "label": label,
        })

    # Determine y-axis ranges — cluster values into up to 3 bands
    all_pred = [v for s in series_data for v in s["pred_means"]]
    all_input = [s["input_mean"] for s in series_data if s["input_mean"] is not None]
    actual_val = next((s["actual_mean"] for s in series_data if s["actual_mean"]), None)

    fig, ax = plt.subplots(figsize=(COL_W, COL_H))

    # Draw prediction curves only (no reference lines, no error bars)
    for s in series_data:
        # Include input ADL value in the legend label
        input_str = f" (input={s['input_mean']:.1f})" if s["input_mean"] is not None else ""
        _plot_series(ax, s["checkpoints"], s["pred_means"], s["pred_errs"],
                     s["color"], s["marker"], "-",
                     f"{s['label']}{input_str}", s["has_multi"], elinewidth=0.8)

    # Actual reference as a single line
    if actual_val is not None:
        ax.axhline(y=actual_val, color="#222222", linestyle="--",
                    linewidth=1.2, label=f"Original ({actual_val:.2f})", zorder=1)

    # Y-limits: cover converged range well, but cap to avoid
    # early-checkpoint spikes stretching the axis too much.
    # Use the 95th percentile of prediction values as the upper limit.
    sorted_pred = sorted(all_pred)
    p95_idx = int(len(sorted_pred) * 0.95)
    y_hi_base = sorted_pred[p95_idx] if sorted_pred else 3.5
    # Ensure we at least cover the actual reference
    if actual_val is not None:
        y_hi_base = max(y_hi_base, actual_val)
    y_lo = min(all_pred) - 0.05 if all_pred else 0
    y_hi = y_hi_base + 0.15
    ax.set_ylim(y_lo, y_hi)

    if show_legend:
        ax.legend(loc="best", ncol=1, frameon=True)
    ax.margins(x=0.02)

    fig.tight_layout()
    _save(fig, output_path)


def plot_dep_f1_with_baseline(csv_grouped, output_path, error_bar="std", corpus_title="",
                              show_legend=False, legend_loc="lower right"):
    """
    Plot dep F1 prediction vs baseline across checkpoints.
    """
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))

    perturbations = sorted(csv_grouped.keys())
    for idx, perturbation in enumerate(perturbations):
        ckpt_data = csv_grouped[perturbation]
        checkpoints = sorted(ckpt_data.keys())
        label = PERTURBATION_LABELS.get(perturbation, perturbation)
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        f1_means, f1_errs = [], []
        baseline_vals = []
        has_multi = False

        for ckpt in checkpoints:
            cols = ckpt_data[ckpt]
            if "dep_f1" in cols:
                mean, std, sem, vmin, vmax = compute_stats(cols["dep_f1"])
                f1_means.append(mean)
                if len(cols["dep_f1"]) > 1:
                    has_multi = True
                    f1_errs.append(std if error_bar == "std" else
                                   sem if error_bar == "sem" else
                                   max(mean - vmin, vmax - mean))
                else:
                    f1_errs.append(0.0)
            if "baseline_f1" in cols:
                baseline_vals.extend(cols["baseline_f1"])

        if baseline_vals:
            baseline_mean = float(np.mean(baseline_vals))
            ax.axhline(y=baseline_mean, color=color, linestyle=":",
                        linewidth=1.0, alpha=0.7,
                        label=f"Baseline {label} ({baseline_mean:.2f})", zorder=1)

        _plot_series(ax, checkpoints, f1_means, f1_errs, color, marker, "-",
                     label, has_multi)

    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    if show_legend:
        ax.legend(loc=legend_loc, ncol=1, frameon=True)
    ax.margins(x=0.02, y=0.05)

    fig.tight_layout()
    _save(fig, output_path)


def plot_precision_recall(csv_grouped, output_path, error_bar="std", corpus_title="",
                          show_legend=False):
    """Plot precision and recall curves."""
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))

    perturbations = sorted(csv_grouped.keys())
    for idx, perturbation in enumerate(perturbations):
        ckpt_data = csv_grouped[perturbation]
        checkpoints = sorted(ckpt_data.keys())
        label = PERTURBATION_LABELS.get(perturbation, perturbation)
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        for metric_col, ls, suffix in [
            ("dep_precision", "-", "Prec."),
            ("dep_recall", "--", "Rec."),
        ]:
            means, errs = [], []
            has_multi = False
            for ckpt in checkpoints:
                cols = ckpt_data[ckpt]
                if metric_col in cols:
                    mean, std, sem, vmin, vmax = compute_stats(cols[metric_col])
                    means.append(mean)
                    if len(cols[metric_col]) > 1:
                        has_multi = True
                        errs.append(std if error_bar == "std" else
                                    sem if error_bar == "sem" else
                                    max(mean - vmin, vmax - mean))
                    else:
                        errs.append(0.0)

            if means:
                ax.plot(checkpoints, means,
                        color=color, marker=marker, linestyle=ls,
                        linewidth=LINE_WIDTH, markersize=7,
                        markeredgewidth=0.8, markeredgecolor=color,
                        markerfacecolor=color,
                        label=f"{label} {suffix}", zorder=3)

                if has_multi and any(e > 0 for e in errs):
                    ax.errorbar(checkpoints, means, yerr=errs,
                                 fmt="none", ecolor=color, elinewidth=1.5,
                                 capsize=4, capthick=1.0, alpha=0.85, zorder=2)

    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    if show_legend:
        ax.legend(loc="lower right", ncol=1, frameon=True)
    ax.margins(x=0.02, y=0.05)

    fig.tight_layout()
    _save(fig, output_path)


def plot_combined_4panel(csv_grouped, output_path, error_bar="std", corpus_title=""):
    """Combined 2x2 panel: ADL, Dep F1, Precision/Recall, ADL Recovery."""
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_W, 5.0))

    perturbations = sorted(csv_grouped.keys())

    def _get_col_series(pert, col):
        ckpt_data = csv_grouped[pert]
        checkpoints = sorted(ckpt_data.keys())
        means, errs = [], []
        has_multi = False
        for ckpt in checkpoints:
            cols = ckpt_data[ckpt]
            if col in cols:
                mean, std, sem, vmin, vmax = compute_stats(cols[col])
                means.append(mean)
                if len(cols[col]) > 1:
                    has_multi = True
                    errs.append(std if error_bar == "std" else
                                sem if error_bar == "sem" else
                                max(mean - vmin, vmax - mean))
                else:
                    errs.append(0.0)
        return checkpoints, means, errs, has_multi

    # (a) ADL
    ax = axes[0, 0]
    first_actual = None
    for idx, perturbation in enumerate(perturbations):
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        label = PERTURBATION_LABELS.get(perturbation, perturbation)
        checkpoints, pred_means, pred_errs, has_multi = _get_col_series(perturbation, "adl_prediction")

        # Reference lines (from all checkpoints)
        ckpt_data = csv_grouped[perturbation]
        actual_refs = [v for ckpt in ckpt_data for v in ckpt_data[ckpt].get("adl_actual", [])]
        input_refs = [v for ckpt in ckpt_data for v in ckpt_data[ckpt].get("adl_input", [])]

        if actual_refs and first_actual is None:
            first_actual = float(np.mean(actual_refs))
            ax.axhline(y=first_actual, color="#222222", linestyle="--",
                        linewidth=1.0, label="Original", zorder=1)
        if input_refs:
            ax.axhline(y=float(np.mean(input_refs)), color=color, linestyle=":",
                        linewidth=0.8, alpha=0.6, label=f"Input {label}", zorder=1)

        _plot_series(ax, checkpoints, pred_means, pred_errs, color, marker, "-",
                     f"Pred. {label}", has_multi)

    ax.set_title(r"$\bf{(a)}$ Average Dependency Length", fontsize=11, loc="left")
    ax.legend(loc="best")

    # (b) Dep F1
    ax = axes[0, 1]
    for idx, perturbation in enumerate(perturbations):
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        label = PERTURBATION_LABELS.get(perturbation, perturbation)
        checkpoints, f1_means, f1_errs, has_multi = _get_col_series(perturbation, "dep_f1")

        baseline_vals = [v for ckpt in csv_grouped[perturbation]
                         for v in csv_grouped[perturbation][ckpt].get("baseline_f1", [])]
        if baseline_vals:
            bm = float(np.mean(baseline_vals))
            ax.axhline(y=bm, color=color, linestyle=":",
                        linewidth=0.8, alpha=0.6,
                        label=f"Baseline {label} ({bm:.2f})", zorder=1)

        _plot_series(ax, checkpoints, f1_means, f1_errs, color, marker, "-",
                     label, has_multi)

    ax.set_title(r"$\bf{(b)}$ Dependency Triple F$_1$", fontsize=11, loc="left")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(loc="lower right")

    # (c) Precision & Recall
    ax = axes[1, 0]
    for idx, perturbation in enumerate(perturbations):
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        label = PERTURBATION_LABELS.get(perturbation, perturbation)

        for col, ls, suffix in [("dep_precision", "-", "Prec."), ("dep_recall", "--", "Rec.")]:
            checkpoints, means, errs, has_multi = _get_col_series(perturbation, col)

            ax.plot(checkpoints, means, color=color, marker=marker, linestyle=ls,
                    linewidth=LINE_WIDTH, markersize=7,
                    markeredgewidth=0.8, markeredgecolor=color,
                    markerfacecolor=color,
                    label=f"{label} {suffix}", zorder=3)
            if has_multi and any(e > 0 for e in errs):
                ax.errorbar(checkpoints, means, yerr=errs,
                             fmt="none", ecolor=color, elinewidth=1.5,
                             capsize=4, capthick=1.0, alpha=0.85, zorder=2)

    ax.set_title(r"$\bf{(c)}$ Precision & Recall", fontsize=11, loc="left")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(loc="lower right")

    # (d) ADL Recovery
    ax = axes[1, 1]
    for idx, perturbation in enumerate(perturbations):
        ckpt_data = csv_grouped[perturbation]
        checkpoints = sorted(ckpt_data.keys())
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        label = PERTURBATION_LABELS.get(perturbation, perturbation)

        recovery = []
        for ckpt in checkpoints:
            cols = ckpt_data[ckpt]
            pred = float(np.mean(cols.get("adl_prediction", [0])))
            inp = float(np.mean(cols.get("adl_input", [0])))
            act = float(np.mean(cols.get("adl_actual", [0])))
            gap = inp - act
            recovered = inp - pred
            recovery.append((recovered / gap * 100) if gap > 0 else 0)

        ax.plot(checkpoints, recovery,
                color=color, marker=marker, linestyle="-",
                linewidth=LINE_WIDTH, markersize=7,
                markeredgewidth=0.8, markeredgecolor=color,
                markerfacecolor=color,
                label=label, zorder=3)

    ax.axhline(y=100, color="#222222", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    ax.set_title(r"$\bf{(d)}$ ADL Gap Recovery", fontsize=11, loc="left")
    ax.legend(loc="lower right")

    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    _save(fig, output_path)


def plot_final_bar_chart(experiments, metric_suffix, metric_label, output_path, error_bar="std"):
    """
    Bar chart of final checkpoint metric across corpora and perturbations.
    """
    grouped = aggregate_metric_across_seeds(experiments, metric_suffix)

    # Reorganize: corpus -> perturbation -> values
    corpus_pert = defaultdict(dict)
    for (corpus, perturbation), ckpt_data in grouped.items():
        # Take the last checkpoint
        last_ckpt = max(ckpt_data.keys())
        corpus_pert[corpus][perturbation] = ckpt_data[last_ckpt]

    if not corpus_pert:
        return

    corpora = sorted(corpus_pert.keys())
    all_perts = sorted({p for perps in corpus_pert.values() for p in perps})

    fig, ax = plt.subplots(figsize=(max(4.8, len(corpora) * 2), 3.6))
    x = np.arange(len(corpora))
    bar_w = 0.8 / max(len(all_perts), 1)

    for idx, pert in enumerate(all_perts):
        means, errs = [], []
        for corpus in corpora:
            vals = corpus_pert[corpus].get(pert, [])
            if vals:
                mean, std, sem, vmin, vmax = compute_stats(vals)
                means.append(mean)
                if error_bar == "std":
                    errs.append(std)
                elif error_bar == "sem":
                    errs.append(sem)
                else:
                    errs.append(0)
            else:
                means.append(0)
                errs.append(0)

        offset = (idx - (len(all_perts) - 1) / 2.0) * bar_w
        ax.bar(x + offset, means, width=bar_w * 0.92,
               color=COLORS[idx % len(COLORS)],
               edgecolor="white", linewidth=0.6,
               yerr=errs, capsize=3, ecolor="0.25",
               alpha=0.9,
               label=PERTURBATION_LABELS.get(pert, pert))

    ax.set_xticks(x)
    ax.set_xticklabels([CORPUS_LABELS.get(c, c) for c in corpora])
    if metric_suffix in {"exact_match", "BLEU", "dep_f1"}:
        ax.set_ylim(0, 1.0)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate all paper plots from evaluation_results.")
    parser.add_argument("--results-dir", default="evaluation_results",
                        help="Root directory containing results.")
    parser.add_argument("--output-dir", default="paper_plots",
                        help="Output directory for plots.")
    parser.add_argument("--error-bar", default="std", choices=["std", "sem", "minmax"],
                        help="Error bar type (default: std = vertical bars with caps).")
    parser.add_argument("--line-width", type=float, default=None,
                        help="Line thickness for all plots (default: 1.8).")
    parser.add_argument("--perturbations", nargs="*", default=None,
                        help="Only include these perturbation types (e.g. fullShuffle localShuffle3 partialReverse).")
    args = parser.parse_args()

    global LINE_WIDTH
    if args.line_width is not None:
        LINE_WIDTH = args.line_width

    setup_acl_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = discover_experiments(args.results_dir)
    # Filter perturbations if requested
    if args.perturbations:
        allowed = set(args.perturbations)
        experiments = [e for e in experiments if e["perturbation"] in allowed]
    print(f"Discovered {len(experiments)} experiment runs:")
    corpus_pert_seeds = defaultdict(lambda: defaultdict(list))
    for exp in experiments:
        corpus_pert_seeds[exp["corpus"]][exp["perturbation"]].append(exp["seed"])
    for corpus, perts in sorted(corpus_pert_seeds.items()):
        for pert, seeds in sorted(perts.items()):
            print(f"  {corpus}/{pert}: {len(seeds)} seeds ({', '.join(sorted(seeds))})")

    # Group experiments by corpus
    by_corpus = defaultdict(list)
    for exp in experiments:
        by_corpus[exp["corpus"]].append(exp)

    # Legend appears inside exactly one plot across all corpora:
    # the dep_f1 plot for the first corpus (alphabetically).
    legend_corpus = sorted(by_corpus.keys())[0] if by_corpus else None

    main_dir = output_dir / "main"
    main_dir.mkdir(parents=True, exist_ok=True)

    for corpus, corpus_exps in sorted(by_corpus.items()):
        corpus_dir = output_dir / corpus
        corpus_dir.mkdir(parents=True, exist_ok=True)
        corpus_label = CORPUS_LABELS.get(corpus, corpus)
        is_legend_corpus = (corpus == legend_corpus)

        print(f"\n{'='*60}")
        print(f"Corpus: {corpus_label}")
        print(f"{'='*60}")

        # 1. EM + BLEU combined checkpoint curves
        print("\n  [EM + BLEU combined]")
        grouped_em = {}
        grouped_bleu = {}
        em_data = aggregate_metric_across_seeds(corpus_exps, "exact_match")
        bleu_data = aggregate_metric_across_seeds(corpus_exps, "BLEU")
        for (c, pert), ckpt_data in em_data.items():
            grouped_em[pert] = ckpt_data
        for (c, pert), ckpt_data in bleu_data.items():
            grouped_bleu[pert] = ckpt_data

        if grouped_em or grouped_bleu:
            plot_em_bleu_combined(
                grouped_em, grouped_bleu,
                main_dir / f"{corpus}_em_bleu.png",
                error_bar=args.error_bar,
                corpus_title=corpus_label,
                show_legend=True,
            )

        # 2. Individual metric checkpoint curves
        for metric_suffix, metric_label in [
            ("exact_match", "Exact Match"),
            ("BLEU", "BLEU"),
            ("dep_f1", r"Dependency F$_1$"),
        ]:
            print(f"\n  [{metric_label}]")
            grouped = aggregate_metric_across_seeds(corpus_exps, metric_suffix)
            pert_data = {pert: ckpt_data for (c, pert), ckpt_data in grouped.items()}
            if pert_data:
                plot_checkpoint_curves(
                    pert_data, metric_label,
                    corpus_dir / f"{corpus}_{metric_suffix}.png",
                    error_bar=args.error_bar,
                    use_cbrt_scale=(metric_suffix in {"exact_match", "BLEU"}),
                    corpus_title=corpus_label,
                )

        # 3. ADL multi-panel (one subplot per perturbation with reference lines)
        print("\n  [ADL multi-panel]")
        # Will be generated after CSV data is loaded below

        # 4. CSV-based plots (ADL with references, dep F1 with baseline, P/R, combined)
        print("\n  [CSV-based dependency analysis]")
        csv_data = aggregate_csv_across_seeds(corpus_exps)
        csv_by_pert = defaultdict(dict)
        for (c, pert), ckpt_dict in csv_data.items():
            for ckpt, cols in ckpt_dict.items():
                csv_by_pert[pert][ckpt] = cols

        if csv_by_pert:
            plot_adl_multipanel(
                csv_by_pert,
                main_dir / f"{corpus}_adl.png",
                error_bar=args.error_bar,
                corpus_title=corpus_label,
                show_legend=True,
            )

            plot_adl_curves(
                csv_by_pert,
                corpus_dir / f"{corpus}_adl_detail",
                error_bar=args.error_bar,
                corpus_title=corpus,
            )

            plot_dep_f1_with_baseline(
                csv_by_pert,
                main_dir / f"{corpus}_dep_f1_baseline.png",
                error_bar=args.error_bar,
                corpus_title=corpus,
                show_legend=True,
                legend_loc="center right" if corpus == "gutenberg" else "lower right",
            )

            plot_precision_recall(
                csv_by_pert,
                corpus_dir / f"{corpus}_precision_recall.png",
                error_bar=args.error_bar,
                corpus_title=corpus,
            )

            plot_combined_4panel(
                csv_by_pert,
                corpus_dir / f"{corpus}_combined_4panel.png",
                error_bar=args.error_bar,
                corpus_title=corpus,
            )

    # 5. Cross-corpus bar charts (final checkpoint)
    print(f"\n{'='*60}")
    print("Cross-corpus bar charts (final checkpoint)")
    print(f"{'='*60}")
    for metric_suffix, metric_label in [
        ("exact_match", "Exact Match"),
        ("BLEU", "BLEU"),
        ("dep_f1", r"Dependency F$_1$"),
    ]:
        plot_final_bar_chart(
            experiments, metric_suffix, metric_label,
            output_dir / f"bar_{metric_suffix}.png",
            error_bar="std",
        )

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
