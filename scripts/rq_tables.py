"""Write LaTeX tables for the arc-distance (slopes) and relation-type (ORs) analyses from rq_v2 outputs."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / 'evaluation_results' / 'rq_v2'
OUT = ROOT / 'report' / 'section'
NAMES = {'partialReverse': r'\textsc{PartialReverse}', 'localShuffle3': r'\textsc{LocalShuffle}',
         'fullShuffle': r'\textsc{FullShuffle}'}
DS = ['bnc_spoken', 'gutenberg']


def orci(b, lo, hi):
    return f'{np.exp(b):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}]'


desc = pd.read_csv(IN / 'arc_descriptives.csv').set_index(['dataset', 'perturbation'])
sl = pd.read_csv(IN / 'rq1_slopes.csv').set_index(['dataset', 'perturbation'])
glmer_path = IN / 'rq1_glmer_slopes.csv'
gl = pd.read_csv(glmer_path).set_index(['dataset', 'perturbation']) if glmer_path.exists() else None

lines = [r'\begin{table*}[t]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{l' + 'rrll' * 2 + '}', r'\toprule',
         r' & \multicolumn{4}{c}{\textsl{bnc\_spoken}} & \multicolumn{4}{c}{\textsl{Gutenberg}} \\',
         r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
         r' & & & \multicolumn{2}{c}{OR per doubling} & & & \multicolumn{2}{c}{OR per doubling} \\',
         r'\cmidrule(lr){4-5}\cmidrule(lr){8-9}',
         r'Perturbation & Rest. & Len. & GLM & Mixed & Rest. & Len. & GLM & Mixed \\', r'\midrule']
for p, name in NAMES.items():
    cells = [name]
    for ds in DS:
        d, s = desc.loc[(ds, p)], sl.loc[(ds, p)]
        cells += [f'{d.recovery:.2f}', f'{d.median_pert_len:.0f}', orci(s.slope, s.lo, s.hi)]
        cells.append(orci(*gl.loc[(ds, p), ['slope', 'lo', 'hi']]) if gl is not None else '--')
    lines.append(' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}',
          r'\caption{\footnotesize{Arc restoration by perturbed arc length. \emph{Rest.}: share of original arcs restored; '
          r'\emph{Len.}: median perturbed arc length (GPT-2 subwords). OR per doubling: odds ratio for restoration per doubling of '
          r'perturbed length, from the distance $\times$ perturbation model (M2), controlling for original arc length and sentence length. '
          r'GLM: cluster-robust 95\% CI by source sentence; mixed: random intercepts for source sentence and model output.}}',
          r'\label{tab:rq1-slopes}', r'\end{table*}']
(OUT / 'tab_rq1.tex').write_text('\n'.join(lines) + '\n')

o = pd.read_csv(IN / 'rq2_relation_ors.csv')
w = o.pivot(index='relation', columns='dataset', values=['OR', 'OR_lo', 'OR_hi', 'n'])
w = w.sort_values(('OR', 'gutenberg'), na_position='last')
lines = [r'\begin{table}[t]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{lrlrl}', r'\toprule',
         r' & \multicolumn{2}{c}{\textsl{bnc\_spoken}} & \multicolumn{2}{c}{\textsl{Gutenberg}} \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}', r'Relation & $n$ & OR [95\% CI] & $n$ & OR [95\% CI] \\', r'\midrule']
for rel, r in w.iterrows():
    cells = [r"\texttt{neg} (\emph{n't})" if rel == 'neg_clitic' else r'\texttt{' + rel + '}']
    for ds in DS:
        if np.isnan(r[('OR', ds)]):
            cells += ['--', '--']
        else:
            cells += [f"{int(r[('n', ds)]):,}".replace(',', '{,}'),
                      f"{r[('OR', ds)]:.2f} [{r[('OR_lo', ds)]:.2f}, {r[('OR_hi', ds)]:.2f}]"]
    lines.append(' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}',
          r'\caption{\footnotesize{Odds ratios for restoration by relation type (reference \texttt{nsubj}), controlling for '
          r'perturbed and original arc length, sentence length, perturbation type and dependent-word frequency; cluster-robust 95\% CIs '
          r'by source sentence. $n$: arcs per training seed. Relations with fewer than 1{,}000 arcs across seeds are omitted (--). '
          r"Negation is split into the clitic \emph{n't}, which shares a GPT-2 subword with its auxiliary (\emph{didn}+\emph{'t}), and free-word negators.}}",
          r'\label{tab:rq2-relations}', r'\end{table}']
(OUT / 'tab_rq2.tex').write_text('\n'.join(lines) + '\n')

q = pd.read_csv(IN / 'rq3_results.csv')
q = q[q.subset == 'all'].set_index(['dataset', 'perturbation'])
lc = pd.read_csv(IN / 'rq3_length_control.csv').set_index(['dataset', 'perturbation'])
lines = [r'\begin{table}[t]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{2.5pt}',
         r'\begin{tabular}{llrlrrrr}', r'\toprule',
         r' & & & & \multicolumn{1}{c}{$\Delta$ADL} & \multicolumn{3}{c}{slope on displacement} \\',
         r'\cmidrule(lr){6-8}',
         r'Data & Pert. & EM & $\Delta$ADL [95\% CI] & same len. & model & $R^2$ & copy \\', r'\midrule']
short = {'partialReverse': r'\textsc{PR}', 'localShuffle3': r'\textsc{LS}', 'fullShuffle': r'\textsc{FS}'}
for ds, dsname in [('bnc_spoken', r'\textsl{bnc}'), ('gutenberg', r'\textsl{Gut.}')]:
    for p in NAMES:
        r, l = q.loc[(ds, p)], lc.loc[(ds, p)]
        star = '$^{*}$' if r.p_disp < 0.05 else ''
        lines.append(' & '.join([dsname if p == 'partialReverse' else '', short[p], f'{r.em:.2f}',
                                 f'{r.mean_gap:+.2f} [{r.gap_lo:+.2f}, {r.gap_hi:+.2f}]', f'{l.gap_same_len:+.2f}',
                                 f'{r.b_disp:+.3f}{star}', f'{r.r2:.3f}', f'{r.b_copy_baseline:+.3f}']) + r' \\')
    if ds == 'bnc_spoken':
        lines.append(r'\midrule')
lines += [r'\bottomrule', r'\end{tabular}',
          r'\caption{\footnotesize{Output dependency length. $\Delta$ADL $=$ ADL$_{\text{pred}}-$ADL$_{\text{orig}}$ in words, '
          r'with 95\% CI over source sentences; \emph{same len.}: $\Delta$ADL among outputs with the original word count. '
          r'Slopes: per-sentence OLS of $\Delta$ADL on standardised input displacement (cluster-robust; $^{*}p<0.05$). '
          r'\emph{copy}: the same slope when the input string itself is taken as the output.}}',
          r'\label{tab:rq3}', r'\end{table}']
(OUT / 'tab_rq3.tex').write_text('\n'.join(lines) + '\n')

sv = pd.read_csv(IN / 'sentences_v2.csv')
per_seed = sv.groupby(['dataset', 'perturbation', 'seed']).agg(
    orig=('gold_adl_w', 'mean'), out=('pred_adl_w', 'mean'), inp=('input_adl_w', 'mean'))
adl = per_seed.groupby(level=[0, 1]).agg(['mean', 'std'])
lines = [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
         r'\begin{tabular}{lcccc}', r'\toprule',
         r' & \multicolumn{2}{c}{\textsl{bnc\_spoken}} & \multicolumn{2}{c}{\textsl{Gutenberg}} \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}', r' & Input & Output & Input & Output \\', r'\midrule',
         r'Original text & \multicolumn{2}{c}{' + f"{adl.loc[('bnc_spoken', 'partialReverse'), ('orig', 'mean')]:.2f}" + '} & '
         r'\multicolumn{2}{c}{' + f"{adl.loc[('gutenberg', 'partialReverse'), ('orig', 'mean')]:.2f}" + r'} \\', r'\midrule']
for p, name in NAMES.items():
    cells = [name]
    for ds in DS:
        r = adl.loc[(ds, p)]
        cells += [f"{r[('inp', 'mean')]:.2f}", f"{r[('out', 'mean')]:.2f} {{\\scriptsize$\\pm$ {r[('out', 'std')]:.2f}}}"]
    lines.append(' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}',
          r'\caption{\footnotesize{Average dependency length (ADL, in words) of the perturbed input and of the model output at the final '
          r'checkpoint, against the original text. Input ADL places the original arcs at their positions in the exact token permutation; '
          r'output ADL is measured on the parse of the output. Output values are means over five training seeds ($\pm$ SD).}}',
          r'\label{tab:adl}', r'\end{table}']
(OUT / 'tab_adl.tex').write_text('\n'.join(lines) + '\n')
print('ok')
