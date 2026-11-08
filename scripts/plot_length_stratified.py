"""
Plot length-stratified EM and BLEU within each corpus, with bootstrap 95% CIs.
Reads: evaluation_results/length_bootstrap/stratified.csv
       evaluation_results/length_bootstrap/overall_bootstrap.csv
Writes: report/figures/length_strat_gutenberg.pdf
        report/figures/length_strat_bnc_spoken.pdf

Usage:
    python scripts/plot_length_stratified.py
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PERT_LABELS = {
    'partialReverse': 'PartialReverse',
    'localShuffle3':  'LocalShuffle',
    'fullShuffle':    'FullShuffle',
    'phraseShuffle':  'PhraseShuf',
}
PERT_ORDER = ['partialReverse', 'localShuffle3', 'phraseShuffle', 'fullShuffle']
COLORS = {
    'partialReverse': '#1f77b4',
    'localShuffle3':  '#ff7f0e',
    'phraseShuffle':  '#2ca02c',
    'fullShuffle':    '#d62728',
}
MARKERS = {
    'partialReverse': 'o',
    'localShuffle3':  's',
    'phraseShuffle':  '^',
    'fullShuffle':    'D',
}

DATASET_TITLES = {
    'gutenberg': 'Gutenberg (long, ≈47 tok)',
    'bnc_spoken': 'BNC Spoken (short, ≈12 tok)',
}


def plot_corpus(df, dset, out_path, metric='em'):
    sub = df[df['dataset'] == dset].copy()
    if sub.empty:
        print(f"No data for {dset}")
        return

    perts_in_data = [p for p in PERT_ORDER if p in sub['perturbation'].unique()]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for pert in perts_in_data:
        g = sub[sub['perturbation'] == pert].sort_values('bin_lo')
        x = g['mean_len'].values
        y = g[f'{metric}_mean'].values
        lo = g[f'{metric}_lo'].values
        hi = g[f'{metric}_hi'].values
        c = COLORS[pert]
        m = MARKERS[pert]

        ax.plot(x, y, color=c, marker=m, label=PERT_LABELS[pert])
        ax.fill_between(x, lo, hi, alpha=0.15, color=c)

    ax.set_xlabel('Mean sentence length (tokens)')
    ylabel = 'Exact Match' if metric == 'em' else 'BLEU'
    ax.set_ylabel(ylabel)
    ax.set_title(DATASET_TITLES.get(dset, dset))
    ax.set_ylim(bottom=0)
    ax.legend(loc='best', frameon=False)
    ax.grid(axis='y', linestyle=':', alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='evaluation_results/length_bootstrap',
                        help='Directory containing stratified.csv from Snellius')
    parser.add_argument('--out-dir', default='report/figures')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strat_path = data_dir / 'stratified.csv'
    if not strat_path.exists():
        print(f"ERROR: {strat_path} not found — copy from Snellius first:")
        print(f"  scp amohammadi@snellius.surf.nl:/gpfs/home2/amohammadi/length_bootstrap/stratified.csv {strat_path}")
        return

    df = pd.read_csv(strat_path)
    print(f"Loaded {len(df)} rows from {strat_path}")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Perturbations: {df['perturbation'].unique()}")

    for dset in ['gutenberg', 'bnc_spoken']:
        for metric in ['em', 'bleu']:
            fname = f'length_strat_{dset}_{metric}.pdf'
            plot_corpus(df, dset, out_dir / fname, metric=metric)

    # Also print overall_bootstrap table if present
    overall_path = data_dir / 'overall_bootstrap.csv'
    if overall_path.exists():
        ov = pd.read_csv(overall_path)
        print("\n=== Overall bootstrap CIs ===")
        print(ov.to_string(index=False))


if __name__ == '__main__':
    main()
