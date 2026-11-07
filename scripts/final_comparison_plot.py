"""Final-checkpoint EM / BLEU / Triple F1 by perturbation, short vs long sentences."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'evaluation_results'
OUT = ROOT / 'report' / 'figures' / 'final_comparison'
PERTS = ['partialReverse', 'localShuffle3', 'fullShuffle']
XLABELS = ['Partial\nReverse', 'Local\nShuffle', 'Full\nShuffle']
SEEDS = ['42', '1337', '2025', '314159', '8675309']
DS = {'bnc_spoken': dict(label='BNC Spoken (short)', color='#5B87C5', ls='-', marker='o', offset=-1),
      'gutenberg': dict(label='Gutenberg (long)', color='#A3264E', ls='--', marker='s', offset=1)}
PANELS = [('exact_match', 'Exact Match', '(higher is better)', 1, (0, 1)),
          ('BLEU', 'BLEU', '(0–100; higher is better)', 100, (0, 100)),
          ('dep_f1', 'Triple F1', '(higher is better)', 1, (0, 1))]

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
                     'font.size': 9, 'axes.edgecolor': '#999999', 'axes.linewidth': 0.6,
                     'xtick.color': '#333333', 'ytick.color': '#555555'})


def load(ds, pert, key, scale):
    vals = []
    for s in SEEDS:
        f = RES / ds / pert / s / f'full_samples_1k_{ds}_{pert}_final_metrics.json'
        vals.append(json.loads(f.read_text())[key] * scale)
    return np.mean(vals), np.std(vals, ddof=1)


base = pd.read_csv(RES / 'rq_v2' / 'input_baselines.csv').set_index(['dataset', 'perturbation'])
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
x = np.arange(len(PERTS))
for ax, (key, title, sub, scale, ylim) in zip(axes, PANELS):
    span = ylim[1] - ylim[0]
    means = {ds: np.array([load(ds, p, key, scale)[0] for p in PERTS]) for ds in DS}
    for ds, st in DS.items():
        m = means[ds]
        other = means['gutenberg' if ds == 'bnc_spoken' else 'bnc_spoken']
        ax.plot(x, m, color=st['color'], ls=st['ls'], marker=st['marker'], ms=5, lw=1.6, label=st['label'], zorder=3)
        for xi, v, o in zip(x, m, other):
            txt = f'{v:.0f}' if scale == 100 else f'{v:.2f}'
            # the higher series is labelled above, the lower below, so labels never collide
            dy = 0.075 * span * (1 if v >= o else -1)
            if v + dy < ylim[0] + 0.05 * span:
                ax.text(xi - 0.1, v, txt, color=st['color'], ha='right', va='center', fontsize=8.5)
            else:
                ax.text(xi, v + dy, txt, color=st['color'], ha='center', va='center', fontsize=8.5)
        if key == 'dep_f1':
            b = [base.loc[(ds, p), 'baseline_f1'] for p in PERTS]
            ax.scatter(x + 0.18 + 0.06 * st['offset'], b, marker=st['marker'], s=18, facecolors='none',
                       edgecolors=st['color'], linewidths=0.8, alpha=0.75, zorder=2)
    ax.set_title(r'$\bf{' + title.replace(' ', r'\ ') + '}$ ' + sub, fontsize=9, pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS)
    ax.set_xlim(-0.35, len(PERTS) - 0.65)
    ax.set_ylim(ylim[0] - 0.03 * span, ylim[1] + 0.03 * span)
    ax.grid(axis='y', color='#E5E5E5', lw=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0, pad=4)

handles, labels = axes[0].get_legend_handles_labels()
h_base = plt.Line2D([], [], marker='o', ls='none', mfc='none', mec='#777777', ms=5)
fig.legend(handles + [h_base], labels + ['Copy-input baseline (F1)'], loc='upper center', ncol=3,
           frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.92))
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}.{ext}', dpi=300, bbox_inches='tight')
print('saved', OUT)
