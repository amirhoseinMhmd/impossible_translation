"""
Parser-free locality measure: mutual information decay with token distance k.
Following the MI-based locality quantification approach (cf. Someya et al. 2025).

MI(k) = sum_{x,y} P(w_i=x, w_{i+k}=y) * log2(P(w_i=x, w_{i+k}=y) / (P(w_i=x) * P(w_{i+k}=y)))

Computed on:
  - 'actual'  texts (original / unperturbed) as baseline
  - 'input'   texts (perturbed) for each perturbation × dataset condition

Pooled across 5 seeds for stability; vocabulary restricted to top-V most frequent
tokens across all texts to reduce sparsity.

Output: /gpfs/home2/amohammadi/mi_locality/
  - mi_by_k.csv        — MI(k) for k=1..15 per condition
  - mi_summary.csv     — decay rate (slope of log MI vs k), integral (sum MI_1..15)
"""
import json
import csv
import numpy as np
from pathlib import Path
from collections import Counter

SEEDS = [42, 1337, 2025, 314159, 8675309]
PERTURBATIONS = {
    'partialReverse': ['gutenberg', 'bnc_spoken'],
    'localShuffle3':  ['gutenberg', 'bnc_spoken'],
    'fullShuffle':    ['gutenberg', 'bnc_spoken'],
    'phraseShuffle':  ['gutenberg'],
}
MODEL_BASE = Path('/gpfs/home2/amohammadi/models')
OUT_DIR    = Path('/gpfs/home2/amohammadi/mi_locality')
OUT_DIR.mkdir(exist_ok=True)

VOCAB_SIZE = 3000
K_MAX = 15


# ── load texts ────────────────────────────────────────────────────────────────
def load_texts(pert, dset, field='input'):
    texts = []
    for seed in SEEDS:
        pat = MODEL_BASE / f'{pert}-{dset}-{seed}' / \
              f'full_samples_1k_{dset}_{pert}_checkpoint-5000.json'
        if not pat.exists():
            continue
        for s in json.load(open(pat)):
            t = s.get(field, '').strip()
            if t:
                texts.append(t.split())
    return texts


# ── build shared vocabulary ───────────────────────────────────────────────────
print("Building vocabulary ...")
all_counts: Counter = Counter()
# add original texts first
pert0 = list(PERTURBATIONS.keys())[0]
dset0 = list(PERTURBATIONS[pert0])[0]
for seed in SEEDS:
    pat = MODEL_BASE / f'{pert0}-{dset0}-{seed}' / \
          f'full_samples_1k_{dset0}_{pert0}_checkpoint-5000.json'
    if not pat.exists():
        continue
    for s in json.load(open(pat)):
        all_counts.update(s.get('actual', '').strip().split())
        all_counts.update(s.get('input',  '').strip().split())

for pert, dsets in PERTURBATIONS.items():
    for dset in dsets:
        for seed in SEEDS:
            pat = MODEL_BASE / f'{pert}-{dset}-{seed}' / \
                  f'full_samples_1k_{dset}_{pert}_checkpoint-5000.json'
            if not pat.exists():
                continue
            for s in json.load(open(pat)):
                all_counts.update(s.get('input', '').strip().split())

vocab = {w: i for i, (w, _) in enumerate(all_counts.most_common(VOCAB_SIZE))}
V = len(vocab)
print(f"Vocabulary size: {V}")


# ── MI computation ────────────────────────────────────────────────────────────
def compute_mi_curve(token_lists, k_max=K_MAX):
    """
    Compute MI(k) for k=1..k_max on a list of token sequences.
    Returns array of MI values (length k_max).
    """
    # unigram counts
    uni = np.zeros(V, dtype=np.float64)
    for toks in token_lists:
        for t in toks:
            if t in vocab:
                uni[vocab[t]] += 1
    total_uni = uni.sum()
    if total_uni == 0:
        return np.zeros(k_max)
    p_uni = uni / total_uni

    mi_vals = []
    for k in range(1, k_max + 1):
        # bigram counts at distance k
        pair_count: Counter = Counter()
        total_pairs = 0
        for toks in token_lists:
            ids = [vocab[t] for t in toks if t in vocab]
            for i in range(len(ids) - k):
                pair_count[(ids[i], ids[i + k])] += 1
                total_pairs += 1
        if total_pairs == 0:
            mi_vals.append(0.0)
            continue

        # MI = sum p(x,y) * log2(p(x,y) / (p(x)*p(y)))
        mi = 0.0
        for (x, y), cnt in pair_count.items():
            pxy = cnt / total_pairs
            denom = p_uni[x] * p_uni[y]
            if denom > 0 and pxy > 0:
                mi += pxy * np.log2(pxy / denom)
        mi_vals.append(max(mi, 0.0))  # clip numerical negatives
    return np.array(mi_vals)


def decay_rate(mi_vals):
    """Fit log(MI_k) ~ intercept + slope*k; return slope (negative = decay)."""
    ks = np.arange(1, len(mi_vals) + 1)
    valid = mi_vals > 1e-9
    if valid.sum() < 3:
        return float('nan')
    slope, _ = np.polyfit(ks[valid], np.log2(mi_vals[valid]), 1)
    return float(slope)


# ── compute for all conditions ────────────────────────────────────────────────
conditions = {}

# original (unperturbed) baseline — use 'actual' from any condition
print("Computing MI for original texts ...")
for dset in ['gutenberg', 'bnc_spoken']:
    # get 'actual' texts from first available perturbation for this dset
    for pert in PERTURBATIONS:
        if dset in PERTURBATIONS[pert]:
            texts = load_texts(pert, dset, field='actual')
            if texts:
                label = f'original/{dset}'
                print(f"  {label}: {len(texts)} sentences", flush=True)
                conditions[label] = texts
                break

# perturbed inputs
for pert, dsets in PERTURBATIONS.items():
    for dset in dsets:
        texts = load_texts(pert, dset, field='input')
        if texts:
            label = f'{pert}/{dset}'
            print(f"  {label}: {len(texts)} sentences", flush=True)
            conditions[label] = texts

ORDER = [
    'original/gutenberg', 'original/bnc_spoken',
    'partialReverse/gutenberg', 'partialReverse/bnc_spoken',
    'localShuffle3/gutenberg',  'localShuffle3/bnc_spoken',
    'phraseShuffle/gutenberg',
    'fullShuffle/gutenberg',    'fullShuffle/bnc_spoken',
]

print("\nComputing MI curves ...")
mi_by_k_rows = []
summary_rows = []

for label in ORDER:
    if label not in conditions:
        continue
    toks = conditions[label]
    print(f"  {label} ...", end='', flush=True)
    mi_vals = compute_mi_curve(toks)
    integral = float(mi_vals.sum())
    dr = decay_rate(mi_vals)
    print(f" integral={integral:.4f}  decay_rate={dr:.4f}")

    pert, dset = label.split('/')
    for k, mv in enumerate(mi_vals, start=1):
        mi_by_k_rows.append({
            'condition': label, 'perturbation': pert, 'dataset': dset,
            'k': k, 'mi': round(float(mv), 6),
        })
    summary_rows.append({
        'condition': label, 'perturbation': pert, 'dataset': dset,
        'n_sentences': len(toks),
        'mi_k1':  round(float(mi_vals[0]), 6),
        'mi_k3':  round(float(mi_vals[2]), 6) if len(mi_vals) > 2 else '',
        'mi_k5':  round(float(mi_vals[4]), 6) if len(mi_vals) > 4 else '',
        'mi_k10': round(float(mi_vals[9]), 6) if len(mi_vals) > 9 else '',
        'integral': round(integral, 6),
        'decay_rate': round(dr, 6),
    })

# ── print ordering table ──────────────────────────────────────────────────────
print("\n=== ORDERING BY INTEGRAL (sum MI_1..15) ===")
for r in sorted(summary_rows, key=lambda x: -x['integral']):
    print(f"  {r['condition']:<35}  integral={r['integral']:.4f}  "
          f"decay={r['decay_rate']:.4f}  MI(1)={r['mi_k1']:.4f}")

# ── write CSVs ────────────────────────────────────────────────────────────────
def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

write_csv(OUT_DIR / 'mi_by_k.csv', mi_by_k_rows,
          ['condition','perturbation','dataset','k','mi'])
print(f"\nWrote {OUT_DIR}/mi_by_k.csv")

write_csv(OUT_DIR / 'mi_summary.csv', summary_rows,
          ['condition','perturbation','dataset','n_sentences',
           'mi_k1','mi_k3','mi_k5','mi_k10','integral','decay_rate'])
print(f"Wrote {OUT_DIR}/mi_summary.csv")

print("Done.")
