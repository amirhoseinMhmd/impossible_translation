"""Robustness of the per-perturbation distance slopes to the alignment audit."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / 'evaluation_results' / 'rq_v2'
PERTS = ['partialReverse', 'localShuffle3', 'fullShuffle']
NAMES = {'partialReverse': r'\textsc{PartialReverse}', 'localShuffle3': r'\textsc{LocalShuffle}',
         'fullShuffle': r'\textsc{FullShuffle}'}
KEY = ['dataset', 'perturbation', 'sent_id', 'arc_idx']

arcs = pd.read_csv(IN / 'arcs_v2.csv', usecols=KEY + ['seed', 'is_recovered', 'pert_arc_len_bpe',
                                                      'gold_arc_len_bpe', 'sent_len_bpe'])
audit = pd.read_csv(IN / 'alignment_audit_arcs.csv')
a = arcs.merge(audit, on=KEY, how='left', validate='many_to_one')
assert a.dist_first_min.notna().all()
a['item'] = a.dataset + ':' + a.sent_id.astype(str)
a['perturbation'] = pd.Categorical(a.perturbation, PERTS)
a['ambiguous'] = ((a.dist_first_min != a.dist_first_max) | (a.capped == 1)).astype(int)
a['multi_piece'] = ((a.dep_pieces > 1) | (a.head_pieces > 1)).astype(int)
a['log_gold'] = np.log2(a.gold_arc_len_bpe.clip(lower=1))
for ds in a.dataset.unique():
    m = a.dataset == ds
    a.loc[m, 'sent_len_z'] = (a.loc[m, 'sent_len_bpe'] - a.loc[m, 'sent_len_bpe'].mean()) / a.loc[m, 'sent_len_bpe'].std()

desc = a[a.seed == a.seed.min()].groupby(['dataset', 'perturbation'], observed=True).agg(
    arcs=('ambiguous', 'size'), ambiguous=('ambiguous', 'mean'), multi_piece=('multi_piece', 'mean'),
    first_eq_nearest=('dist_nearest', lambda s: np.nan)).reset_index()
eq = a[a.seed == a.seed.min()]
desc['first_eq_nearest'] = eq.groupby(['dataset', 'perturbation'], observed=True).apply(
    lambda g: (g.pert_arc_len_bpe == g.dist_nearest).mean()).values
sents = pd.read_csv(IN / 'alignment_audit_sentences.csv')
amb_s = sents.groupby(['dataset', 'perturbation']).apply(lambda g: (g.n_solutions > 1).mean()).rename('ambiguous_sentences')
desc = desc.merge(amb_s.reset_index(), on=['dataset', 'perturbation'])
desc.to_csv(IN / 'alignment_audit_summary.csv', index=False)

VARIANTS = [('all', 'pert_arc_len_bpe', None), ('unambiguous', 'pert_arc_len_bpe', 'ambiguous'),
            ('single_subword', 'pert_arc_len_bpe', 'multi_piece'), ('nearest_anchor', 'dist_nearest', None),
            ('last_anchor', 'dist_last', None)]
rows = []
for ds, d in a.groupby('dataset'):
    for name, dist, drop in VARIANTS:
        x = d if drop is None else d[d[drop] == 0]
        x = x.assign(log_dist=np.log2(x[dist].clip(lower=1)))
        m = smf.logit('is_recovered ~ log_dist * C(perturbation) + log_gold + sent_len_z', data=x).fit(
            disp=0, maxiter=200, cov_type='cluster', cov_kwds={'groups': pd.factorize(x['item'])[0]})
        cov = m.cov_params()
        for p in PERTS:
            names = ['log_dist'] + ([f'log_dist:C(perturbation)[T.{p}]'] if p != PERTS[0] else [])
            b = sum(m.params[n] for n in names)
            se = np.sqrt(sum(cov.loc[i, j] for i in names for j in names))
            rows.append(dict(dataset=ds, variant=name, perturbation=p, n=int((x.perturbation == p).sum()),
                             OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se)))
res = pd.DataFrame(rows)
res.to_csv(IN / 'alignment_robustness.csv', index=False)

labels = {'all': 'All arcs (main analysis)', 'unambiguous': 'Unambiguous positions only',
          'single_subword': 'Single-subword endpoints only', 'nearest_anchor': 'Nearest-subword anchor',
          'last_anchor': 'Last-subword anchor'}
lines = [r'\begin{table*}[t]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{4pt}',
         r'\begin{tabular}{l' + 'l' * 6 + '}', r'\toprule',
         r' & \multicolumn{3}{c}{\textsl{bnc\_spoken}} & \multicolumn{3}{c}{\textsl{Gutenberg}} \\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
         'Variant & ' + ' & '.join([NAMES[p] for p in PERTS] * 2) + r' \\', r'\midrule']
for v, lab in labels.items():
    cells = [lab]
    for ds in ['bnc_spoken', 'gutenberg']:
        for p in PERTS:
            r = res[(res.dataset == ds) & (res.variant == v) & (res.perturbation == p)].iloc[0]
            cells.append(f'{r.OR:.2f} [{r.lo:.2f}, {r.hi:.2f}]')
    lines.append(' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}',
          r'\caption{\footnotesize{Robustness of the arc-distance slopes to the position measurement. Odds ratio for restoration per '
          r'doubling of perturbed arc length from the distance $\times$ perturbation model (M2; cluster-robust 95\% CI). '
          r'\emph{Unambiguous}: drops arcs whose perturbed distance differs between \textsc{LocalShuffle} window orders that all reproduce '
          r'the stored input. \emph{Single-subword}: drops arcs whose dependent or head spans more than one GPT-2 subword. '
          r'\emph{Nearest}/\emph{last anchor}: measures distance between the closest pieces, or the last pieces, of the two words instead '
          r'of their first pieces.}}', r'\label{tab:alignment-robustness}', r'\end{table*}']
(ROOT / 'report' / 'section' / 'tab_alignment.tex').write_text('\n'.join(lines) + '\n')
print(desc.round(4).to_string())
print(res.round(3).to_string())
